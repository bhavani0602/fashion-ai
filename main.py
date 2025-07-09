from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum
import uvicorn
import numpy as np
import cv2
import mediapipe as mp
import uuid
import logging
from math import sqrt

app = FastAPI(
    title="Fashion AI Analyzer",
    description="Analyzes body measurements with gender-specific accuracy",
    version="3.3"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Gender(str, Enum):
    female = "female"
    male = "male"

class BodyMeasurements(BaseModel):
    shoulder_width: float
    bust_chest_width: float
    waist_width: float
    hip_width: float
    height: float
    inseam: float
    session_id: str

class FashionResponse(BaseModel):
    gender: str
    body_type: str
    body_description: str
    measurements: BodyMeasurements
    recommend: List[str]
    avoid: List[str]

class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        # Standard measurement ratios based on anthropometric data
        self.standard_ratios = {
            'female': {
                'shoulder_hip': 0.85,
                'waist_hip': 0.75,
                'inseam_height': 0.45
            },
            'male': {
                'shoulder_hip': 1.0,
                'waist_hip': 0.85,
                'inseam_height': 0.47
            }
        }

    def create_session(self, image_data: bytes, gender: Gender, user_height: Optional[float] = None):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'image_data': image_data,
            'gender': gender,
            'user_height': user_height,
            'processed': False
        }
        logger.info(f"Created session: {session_id}")
        return session_id

    def calculate_distance(self, point1, point2, width, height, pixel_to_cm):
        """Calculate Euclidean distance between two landmarks in cm."""
        if point1 is None or point2 is None:
            return None
        dx = (point1.x - point2.x) * width
        dy = (point1.y - point2.y) * height
        return sqrt(dx**2 + dy**2) * pixel_to_cm

    def get_visible_landmark(self, landmarks, landmark_id):
        """Return landmark if visible, otherwise None"""
        landmark = landmarks[landmark_id]
        return landmark if landmark.visibility > 0.5 else None

    def get_pixel_to_cm_ratio(self, landmarks, img_height, user_height=None):
        """Calculate accurate pixel-to-cm conversion based on height landmarks."""
        # Use the vertical distance between nose and ankles as height reference
        nose = self.get_visible_landmark(landmarks, self.mp_pose.PoseLandmark.NOSE)
        left_ankle = self.get_visible_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE)
        right_ankle = self.get_visible_landmark(landmarks, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        
        if not nose:
            raise ValueError("Nose not visible - face must be visible for accurate measurements")
        
        # Use average of both ankles if visible
        ankle_y = None
        if left_ankle and right_ankle:
            ankle_y = (left_ankle.y + right_ankle.y) / 2
        elif left_ankle:
            ankle_y = left_ankle.y
        elif right_ankle:
            ankle_y = right_ankle.y
        
        if ankle_y is None:
            if user_height:
                # If user provided height but we can't detect ankles, use standard ratio
                return 0.75
            raise ValueError("Ankles not visible - stand with feet visible for accurate measurements")
        
        pixel_height = abs(nose.y - ankle_y) * img_height
        
        if user_height:
            # Use user-provided height if available
            return user_height / pixel_height
        else:
            # Fallback to standard ratio if no height provided
            return 0.75

    def process_session(self, session_id: str):
        if session_id not in self.sessions:
            logger.error(f"Session {session_id} not found")
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = self.sessions[session_id]
        if session['processed']:
            logger.info(f"Returning cached result for session: {session_id}")
            return session['result']
        
        try:
            # Decode and process image
            img = cv2.imdecode(np.frombuffer(session['image_data'], np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image format")
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.pose.process(img_rgb)
            
            if not results.pose_landmarks:
                logger.error(f"No pose detected in session {session_id}")
                raise HTTPException(status_code=400, detail="No human pose detected in the image")

            landmarks = results.pose_landmarks.landmark
            height, width, _ = img.shape

            # Calculate pixel-to-cm ratio
            try:
                pixel_to_cm = self.get_pixel_to_cm_ratio(landmarks, height, session['user_height'])
            except ValueError as e:
                logger.warning(f"Using fallback pixel ratio: {str(e)}")
                pixel_to_cm = 0.75  # Fallback value

            # Get all required landmarks
            left_shoulder = self.get_visible_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
            right_shoulder = self.get_visible_landmark(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
            left_hip = self.get_visible_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP)
            right_hip = self.get_visible_landmark(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP)
            nose = self.get_visible_landmark(landmarks, self.mp_pose.PoseLandmark.NOSE)
            left_ankle = self.get_visible_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE)
            right_ankle = self.get_visible_landmark(landmarks, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
            
            # Calculate actual height from image
            actual_height_cm = None
            if nose and (left_ankle or right_ankle):
                ankle_y = None
                if left_ankle and right_ankle:
                    ankle_y = (left_ankle.y + right_ankle.y) / 2
                elif left_ankle:
                    ankle_y = left_ankle.y
                else:
                    ankle_y = right_ankle.y
                
                pixel_height = abs(nose.y - ankle_y) * height
                actual_height_cm = pixel_height * pixel_to_cm
            elif session['user_height']:
                actual_height_cm = session['user_height']
            else:
                actual_height_cm = 170 if session['gender'] == Gender.male else 160

            # Calculate measurements with fallbacks
            shoulder_width = self.calculate_distance(left_shoulder, right_shoulder, width, height, pixel_to_cm)
            
            # Bust/Chest measurement - different approach for male/female
            if session['gender'] == Gender.female:
                # For females, use underbust approximation
                left_underbust = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_underbust = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                bust_chest_width = self.calculate_distance(left_underbust, right_underbust, width, height, pixel_to_cm) * 1.1
            else:
                # For males, use shoulder width with adjustment
                bust_chest_width = shoulder_width * 0.95 if shoulder_width else None
            
            # Waist measurement - use narrowest point between shoulders and hips
            waist_width = self.calculate_distance(left_hip, right_hip, width, height, pixel_to_cm) * 0.85
            
            # Hip measurement
            hip_width = self.calculate_distance(left_hip, right_hip, width, height, pixel_to_cm)
            
            # Inseam measurement
            inseam = None
            if left_hip and left_ankle:
                inseam = self.calculate_distance(left_hip, left_ankle, width, height, pixel_to_cm)
            elif right_hip and right_ankle:
                inseam = self.calculate_distance(right_hip, right_ankle, width, height, pixel_to_cm)
            else:
                # Fallback to standard ratio if can't measure
                inseam = actual_height_cm * self.standard_ratios[session['gender'].value]['inseam_height']

            # Validate measurements
            if None in [shoulder_width, bust_chest_width, waist_width, hip_width, actual_height_cm, inseam]:
                raise HTTPException(status_code=400, detail="Could not detect all required body points. Please ensure full body is visible in the image.")

            measurements = {
                'shoulder_width': round(shoulder_width, 1),
                'bust_chest_width': round(bust_chest_width, 1),
                'waist_width': round(waist_width, 1),
                'hip_width': round(hip_width, 1),
                'height': round(actual_height_cm, 1),
                'inseam': round(inseam, 1),
            }
            
            logger.info(f"Calculated measurements for session {session_id}: {measurements}")

            # Classify body type
            if session['gender'] == Gender.female:
                body_type, description, recommend, avoid = self._classify_female(
                    measurements['shoulder_width'],
                    measurements['bust_chest_width'],
                    measurements['waist_width'],
                    measurements['hip_width']
                )
            else:
                body_type, description, recommend, avoid = self._classify_male(
                    measurements['shoulder_width'],
                    measurements['bust_chest_width'],
                    measurements['waist_width'],
                    measurements['hip_width']
                )

            result = FashionResponse(
                gender=session['gender'].value,
                body_type=body_type,
                body_description=description,
                measurements=BodyMeasurements(
                    **measurements,
                    session_id=session_id
                ),
                recommend=recommend,
                avoid=avoid
            )

            session['result'] = result
            session['processed'] = True
            logger.info(f"Session {session_id} processed successfully")
            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Processing error for session {session_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    def _classify_female(self, shoulder: float, bust: float, waist: float, hip: float):
        ratios = {
            'shoulder_hip': shoulder / hip,
            'bust_waist': bust / waist,
            'waist_hip': waist / hip
        }

        if ratios['shoulder_hip'] < 0.9 and ratios['waist_hip'] < 0.75:
            return (
                "Pear",
                "Wider hips than shoulders with a defined waist",
                ["A-line skirts", "Empire waist tops", "Flared pants"],
                ["Skinny jeans", "Tight crop tops", "Short jackets"]
            )
        elif ratios['shoulder_hip'] > 1.1 and ratios['bust_waist'] > 1.2:
            return (
                "Inverted Triangle",
                "Broad shoulders and bust compared to hips",
                ["V-neck tops", "A-line dresses", "Wide-leg trousers"],
                ["Shoulder pads", "Puffy sleeves", "Turtlenecks"]
            )
        elif 0.95 <= ratios['shoulder_hip'] <= 1.05 and ratios['waist_hip'] < 0.75:
            return (
                "Hourglass",
                "Balanced shoulders and hips with a narrow waist",
                ["Wrap dresses", "Fitted jackets", "High-waisted skirts"],
                ["Baggy outfits", "Boxy tops", "Shapeless dresses"]
            )
        elif ratios['shoulder_hip'] > 0.9 and ratios['waist_hip'] > 0.85:
            return (
                "Apple",
                "Rounded midsection with similar shoulder and hip width",
                ["Tunics", "Empire line dresses", "Straight-leg jeans"],
                ["Tight belts", "Crop tops", "Clingy fabrics"]
            )
        else:
            return (
                "Rectangle",
                "Similar width across shoulders, waist, and hips",
                ["Peplum tops", "Layered looks", "Belted coats"],
                ["Straight shifts", "Boxy jackets", "Oversized tees"]
            )

    def _classify_male(self, shoulder: float, chest: float, waist: float, hip: float):
        ratios = {
            'shoulder_hip': shoulder / hip,
            'chest_waist': chest / waist,
            'waist_hip': waist / hip
        }

        if ratios['shoulder_hip'] > 1.15 and ratios['chest_waist'] > 1.25:
            return (
                "Inverted Triangle",
                "Broad shoulders and chest with narrower waist and hips",
                ["Tailored suits", "Fitted shirts", "Slim jeans"],
                ["Baggy tops", "High-waisted pants", "Turtlenecks"]
            )
        elif 0.95 <= ratios['shoulder_hip'] <= 1.1 and ratios['waist_hip'] < 0.9:
            return (
                "Trapezoid",
                "Athletic build with balanced shoulders and hips",
                ["Slim-fit polos", "Bomber jackets", "Tapered trousers"],
                ["Oversized tees", "Pleated pants", "Loose jackets"]
            )
        elif ratios['waist_hip'] > 0.95 and ratios['chest_waist'] < 1.1:
            return (
                "Oval",
                "Wider midsection with rounded proportions",
                ["Dark shirts", "Vertical stripes", "Structured blazers"],
                ["Tight tees", "Horizontal stripes", "Skinny jeans"]
            )
        else:
            return (
                "Rectangle",
                "Even width across shoulders, waist, and hips",
                ["Layered outfits", "Textured sweaters", "Fitted blazers"],
                ["Baggy clothes", "Shapeless tees", "Wide pants"]
            )

    def close_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Closed session: {session_id}")
        else:
            logger.warning(f"Attempted to close non-existent session: {session_id}")

session_manager = SessionManager()

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    gender: Gender = Form(...),
    user_height: Optional[float] = Form(None)
):
    try:
        logger.info(f"Received upload request - File: {file.filename}, Gender: {gender}, Height: {user_height}cm")
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
        logger.info(f"File size: {len(contents)} bytes")
        session_id = session_manager.create_session(contents, gender, user_height)
        result = session_manager.process_session(session_id)
        return result
    except HTTPException as e:
        logger.error(f"Upload error: {e.status_code}: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/close-session/{session_id}")
async def close_session(session_id: str):
    session_manager.close_session(session_id)
    return {"status": "session closed"}

@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)