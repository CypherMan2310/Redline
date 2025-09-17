# -*- coding: utf-8 -*-
"""
Enhanced WhatsApp Bot with Better Medical Query Handling
"""

import os
import logging
import torch
import re
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, Depends
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from twilio.twiml.messaging_response import MessagingResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn


# ========================
# Enhanced Configuration
# ========================
class Config :
    # Model Configuration
    MODEL_PATH = os.getenv( "MODEL_PATH", "C:/Users/somas/PycharmProjects/Redline/model/fine_tuned_indicbert" )
    MAX_LENGTH = int( os.getenv( "MAX_LENGTH", "128" ) )
    CONFIDENCE_THRESHOLD = float( os.getenv( "CONFIDENCE_THRESHOLD", "0.6" ) )  # Lowered threshold
    LOW_CONFIDENCE_THRESHOLD = float( os.getenv( "LOW_CONFIDENCE_THRESHOLD", "0.3" ) )
    DEVICE = os.getenv( "DEVICE", "cuda" if torch.cuda.is_available() else "cpu" )

    # API Configuration
    HOST = os.getenv( "HOST", "0.0.0.0" )
    PORT = int( os.getenv( "PORT", "8000" ) )
    DEBUG = os.getenv( "DEBUG", "False" ).lower() == "true"

    # Logging
    LOG_LEVEL = os.getenv( "LOG_LEVEL", "INFO" )
    LOG_FILE = os.getenv( "LOG_FILE", "whatsapp_bot.log" )


# Setup logging
logging.basicConfig(
    level=getattr( logging, Config.LOG_LEVEL ),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler( Config.LOG_FILE ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger( __name__ )


# ========================
# Medical Query Detector
# ========================
class MedicalQueryDetector :
    def __init__ ( self ) :
        # Medical keywords and patterns
        self.medical_keywords = {
            'medications' : ['medicine', 'medication', 'drug', 'tablet', 'capsule', 'pill', 'syrup', 'injection'],
            'conditions' : ['disease', 'condition', 'syndrome', 'disorder', 'infection', 'pain', 'fever'],
            'symptoms' : ['symptom', 'side effect', 'reaction', 'allergy', 'dose', 'dosage'],
            'medical_terms' : ['prescribed', 'treatment', 'therapy', 'diagnosis', 'doctor', 'physician'],
            'body_parts' : ['head', 'heart', 'lung', 'liver', 'kidney', 'stomach', 'skin', 'eye', 'ear']
        }

        # Common medication names (you can expand this list)
        self.medication_names = [
            'nystatin', 'ibuprofen', 'paracetamol', 'aspirin', 'amoxicillin',
            'metformin', 'insulin', 'warfarin', 'lisinopril', 'atorvastatin'
        ]

        # Medical question patterns
        self.medical_patterns = [
            r'\b(what is .+ prescribed for)\b',
            r'\b(side effects? of .+)\b',
            r'\b(how to take .+)\b',
            r'\b(dosage of .+)\b',
            r'\b(is .+ safe)\b',
            r'\b(can I take .+)\b'
        ]

    def is_medical_query ( self, text: str ) -> Tuple[bool, str] :
        """Detect if the query is medical and return the type"""
        text_lower = text.lower()

        # Check for medication names
        for med in self.medication_names :
            if med in text_lower :
                return True, "medication_inquiry"

        # Check for medical patterns
        for pattern in self.medical_patterns :
            if re.search( pattern, text_lower ) :
                return True, "medical_question"

        # Check for medical keywords
        medical_score = 0
        total_categories = len( self.medical_keywords )

        for category, keywords in self.medical_keywords.items() :
            if any( keyword in text_lower for keyword in keywords ) :
                medical_score += 1

        # If more than half categories match, it's likely medical
        if medical_score >= total_categories * 0.4 :
            return True, "general_medical"

        return False, "non_medical"


# ========================
# Enhanced Response Generator
# ========================
class EnhancedResponseGenerator :
    def __init__ ( self ) :
        self.medical_detector = MedicalQueryDetector()

        # Enhanced response templates
        self.intent_responses = {
            "complaint" : {
                "high" : "I understand your concern. I've logged your complaint with reference ID: {ref_id}. Our support team will contact you within 24 hours to resolve this issue.",
                "medium" : "It seems you have a concern. Could you please provide more specific details so I can better categorize and forward your complaint to the right department?",
                "low" : "I sense you might have an issue. Could you please rephrase your message or describe your concern in more detail?"
            },
            "inquiry" : {
                "high" : "Thank you for your inquiry! I've categorized your question and our expert team will provide you with comprehensive information shortly.",
                "medium" : "I believe you're seeking information. Let me connect you with our knowledge specialists who can provide detailed answers.",
                "low" : "I think you're asking for information, but I'd like to make sure I understand correctly. Could you please be more specific about what you'd like to know?"
            },
            "drug" : {
                "high" : "I see you're asking about medication. For accurate medical information and safety, I'm connecting you with our healthcare professionals who can provide proper guidance.",
                "medium" : "Your question appears to be about medication or health. For your safety, I recommend consulting with a healthcare professional or pharmacist.",
                "low" : "This seems to be a health-related question. For accurate medical advice, please consult with a qualified healthcare provider."
            },
            "medical" : {
                "high" : "This is a medical question that requires professional expertise. I'm forwarding your query to our healthcare team for a proper response.",
                "medium" : "I recognize this as a health-related inquiry. For accurate information, please consult with a healthcare professional.",
                "low" : "This appears to be a medical question. For your safety and accurate information, please speak with a doctor or pharmacist."
            },
            "greeting" : {
                "high" : "Hello! 👋 Welcome to our healthcare support. How can I assist you today?",
                "medium" : "Hi there! What can I help you with today?",
                "low" : "Hello! How may I help you?"
            },
            "request" : {
                "high" : "I've received your request (ID: {ref_id}) and it's been forwarded to the appropriate team. You'll receive an update soon!",
                "medium" : "I understand you're making a request. I'll ensure the right team handles this promptly.",
                "low" : "It seems like you're requesting something. Could you please provide more details?"
            }
        }

        # Medical-specific responses
        self.medical_responses = {
            "medication_inquiry" : (
                "I understand you're asking about medication. For accurate information about prescriptions, "
                "dosages, and side effects, I strongly recommend consulting with:\n\n"
                "• Your prescribing doctor\n"
                "• A licensed pharmacist\n"
                "• Your healthcare provider\n\n"
                "They can provide safe, personalized medical advice based on your specific situation."
            ),
            "medical_question" : (
                "This is an important medical question that requires professional expertise. "
                "For your safety and to get accurate information, please consult with a qualified healthcare professional. "
                "I've noted your inquiry and can connect you with our medical support team if available."
            ),
            "general_medical" : (
                "I recognize this as a health-related question. While I can help with general inquiries, "
                "medical questions require professional guidance. Please consult with a healthcare provider "
                "for accurate, personalized advice."
            )
        }

    def generate_response ( self, prediction: Dict, user_input: str, user_id: str = None ) -> str :
        """Generate enhanced contextual response"""
        intent = prediction["intent"]
        confidence = prediction["confidence"]

        # Check if it's a medical query first
        is_medical, medical_type = self.medical_detector.is_medical_query( user_input )

        if is_medical :
            logger.info( f"Medical query detected: {medical_type} for user {user_id}" )
            response = self.medical_responses.get( medical_type, self.medical_responses["general_medical"] )

            # Add confidence info for transparency
            if confidence < 0.5 :
                response += f"\n\n_Note: My classification confidence is {confidence * 100:.1f}%, so please verify with a professional._"

            return response

        # Handle low confidence cases
        if confidence < Config.LOW_CONFIDENCE_THRESHOLD :
            return (
                "I'm having difficulty understanding your message clearly. "
                "Could you please rephrase or provide more context? "
                "Alternatively, I can connect you with a human agent who can better assist you.\n\n"
                f"_Confidence: {confidence * 100:.1f}%_"
            )

        # Determine confidence level for non-medical queries
        if confidence >= Config.CONFIDENCE_THRESHOLD :
            conf_level = "high"
        elif confidence >= 0.4 :
            conf_level = "medium"
        else :
            conf_level = "low"

        # Generate reference ID for complaints and requests
        ref_id = f"WA{datetime.now().strftime( '%Y%m%d%H%M%S' )}"

        # Get appropriate response template
        intent_key = intent.lower()
        if intent_key in self.intent_responses :
            response_template = self.intent_responses[intent_key][conf_level]
        else :
            # Default responses for unknown intents
            if conf_level == "high" :
                response_template = f"I've categorized your message as '{intent}' and will ensure the appropriate team handles it."
            elif conf_level == "medium" :
                response_template = f"I think your message relates to '{intent}'. Let me connect you with someone who can help."
            else :
                response_template = "I'm not entirely sure how to categorize your message. Let me connect you with a human agent."

        # Format the response
        response = response_template.format(
            intent=intent,
            confidence=f"{confidence * 100:.1f}%",
            ref_id=ref_id,
            user_input=user_input
        )

        # Add confidence info for medium/low confidence
        if conf_level in ["medium", "low"] :
            response += f"\n\n_Confidence: {confidence * 100:.1f}%_"

        # Log the interaction
        logger.info(
            f"User {user_id}: '{user_input}' -> Intent: {intent} (confidence: {confidence:.3f}, medical: {is_medical})" )

        return response


# ========================
# Model Manager (Same as before but with adjusted thresholds)
# ========================
class IndicBERTModelManager :
    def __init__ ( self ) :
        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.device = Config.DEVICE
        self.model_loaded = False

    async def load_model ( self ) -> bool :
        """Load the IndicBERT model asynchronously"""
        try :
            logger.info( f"Loading model from {Config.MODEL_PATH}" )

            model_path = Path( Config.MODEL_PATH )
            if not model_path.exists() :
                logger.error( f"Model path does not exist: {Config.MODEL_PATH}" )
                return False

            logger.info( "Loading tokenizer..." )
            self.tokenizer = AutoTokenizer.from_pretrained( Config.MODEL_PATH )

            logger.info( f"Loading model on device: {self.device}" )
            self.model = AutoModelForSequenceClassification.from_pretrained( Config.MODEL_PATH )
            self.model.to( self.device )
            self.model.eval()

            self.id2label = self.model.config.id2label
            self.model_loaded = True

            logger.info( f"Model loaded successfully! Available labels: {list( self.id2label.values() )}" )
            return True

        except Exception as e :
            logger.error( f"Failed to load model: {str( e )}" )
            self.model_loaded = False
            return False

    def predict ( self, text: str, return_all_probs: bool = False ) -> Dict :
        """Make prediction with enhanced error handling"""
        if not self.model_loaded :
            raise HTTPException( status_code=500, detail="Model not loaded" )

        try :
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=Config.MAX_LENGTH
            )

            inputs = {k : v.to( self.device ) for k, v in inputs.items()}

            # Make prediction
            with torch.no_grad() :
                outputs = self.model( **inputs )
                probs = torch.nn.functional.softmax( outputs.logits, dim=-1 )
                pred_class = torch.argmax( probs, dim=-1 ).item()
                confidence = probs[0][pred_class].item()

            intent = self.id2label[pred_class]

            result = {
                "intent" : intent,
                "confidence" : confidence,
                "text" : text,
                "timestamp" : datetime.now().isoformat()
            }

            if return_all_probs :
                all_probs = {
                    self.id2label[i] : prob.item()
                    for i, prob in enumerate( probs[0] )
                }
                result["all_probabilities"] = dict( sorted( all_probs.items(), key=lambda x : x[1], reverse=True ) )

            return result

        except Exception as e :
            logger.error( f"Prediction error: {str( e )}" )
            raise HTTPException( status_code=500, detail=f"Prediction failed: {str( e )}" )


# ========================
# FastAPI Application
# ========================
app = FastAPI(
    title="Enhanced WhatsApp Medical Bot",
    description="FastAPI WhatsApp bot with enhanced medical query handling",
    version="2.1.0"
)

app.add_middleware( CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                    allow_headers=["*"] )

# Initialize components
model_manager = IndicBERTModelManager()
response_generator = EnhancedResponseGenerator()


@app.on_event( "startup" )
async def startup_event () :
    """Load model on startup"""
    logger.info( "Starting Enhanced WhatsApp Medical Bot..." )
    success = await model_manager.load_model()
    if not success :
        logger.error( "Failed to load model on startup!" )


@app.post( "/webhook", response_class=PlainTextResponse )
async def whatsapp_webhook (
        From: str = Form( ... ),
        Body: str = Form( ... ),
        MessageSid: Optional[str] = Form( None )
) :
    """Enhanced Twilio WhatsApp webhook"""
    try :
        user_message = Body.strip()
        user_phone = From

        logger.info( f"Received from {user_phone}: '{user_message}'" )

        if not user_message :
            response = MessagingResponse()
            response.message( "I didn't receive any text. Please send your message again." )
            return PlainTextResponse( str( response ), media_type="application/xml" )

        # Handle help commands
        if user_message.lower() in ["help", "/help", "menu", "/menu"] :
            help_text = (
                "🤖 *Medical Support Bot*\n\n"
                "I can help you with:\n"
                "• General health inquiries\n"
                "• Medication questions (with professional guidance)\n"
                "• Medical appointment requests\n"
                "• Health-related complaints\n\n"
                "⚠️ *Important*: For medical advice, always consult healthcare professionals.\n\n"
                "Send me your message and I'll categorize it appropriately!"
            )
            response = MessagingResponse()
            response.message( help_text )
            return PlainTextResponse( str( response ), media_type="application/xml" )

        # Make prediction
        try :
            prediction = model_manager.predict( user_message, return_all_probs=False )
        except Exception as e :
            logger.error( f"Prediction failed for {user_phone}: {str( e )}" )
            response = MessagingResponse()
            response.message( "I'm experiencing technical difficulties. Please try again or contact support directly." )
            return PlainTextResponse( str( response ), media_type="application/xml" )

        # Generate enhanced response
        reply = response_generator.generate_response( prediction, user_message, user_phone )

        response = MessagingResponse()
        response.message( reply )

        logger.info( f"Sent to {user_phone}: Intent={prediction['intent']}, Confidence={prediction['confidence']:.3f}" )

        return PlainTextResponse( str( response ), media_type="application/xml" )

    except Exception as e :
        logger.error( f"Webhook error: {str( e )}" )
        response = MessagingResponse()
        response.message( "I encountered an error. Please try again or contact support." )
        return PlainTextResponse( str( response ), media_type="application/xml" )


# Health check and other endpoints remain the same...
@app.get( "/health" )
async def health_check () :
    return {
        "status" : "healthy" if model_manager.model_loaded else "unhealthy",
        "model_loaded" : model_manager.model_loaded,
        "device" : Config.DEVICE,
        "confidence_threshold" : Config.CONFIDENCE_THRESHOLD,
        "low_confidence_threshold" : Config.LOW_CONFIDENCE_THRESHOLD,
        "timestamp" : datetime.now().isoformat()
    }


if __name__ == "__main__" :
    logger.info( f"Starting server on {Config.HOST}:{Config.PORT}" )
    uvicorn.run( app, host=Config.HOST, port=Config.PORT )