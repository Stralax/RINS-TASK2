import os
import torch
from PIL import Image
import speech_recognition as sr
from gtts import gTTS
import pygame
import time
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import argparse
import cv2
import numpy as np

# Updated list with only the specific birds
COMMON_BIRDS = [
    "002.laysan_albatross", "012.yellow_headed_blackbird", "014.indigo_bunting",
    "025.pelagic_cormorant", "029.american_crow", "033.yellow_billed_cuckoo",
    "035.purple_finch", "042.vermilion_flycatcher", "048.european_goldfinch",
    "050.eared_grebe", "059.california_gull", "068.ruby_throated_hummingbird",
    "073.blue_jay", "081.pied_kingfisher", "095.baltimore_oriole",
    "101.white_pelican", "106.horned_puffin", "108.white_necked_raven",
    "112.great_grey_shrike", "118.house_sparrow", "134.cape_glossy_starling",
    "138.tree_swallow", "144.common_tern", "191.red_headed_woodpecker"
]

# Cleaned bird names for spoken dialogue
BIRD_DISPLAY_NAMES = {
    "002.Laysan_Albatross": "Laysan Albatross",
    "012.Yellow_headed_Blackbird": "Yellow-headed Blackbird",
    "014.Indigo_Bunting": "Indigo Bunting",
    "025.Pelagic_Cormorant": "Pelagic Cormorant",
    "029.American_Crow": "American Crow",
    "033.Yellow_billed_Cuckoo": "Yellow-billed Cuckoo",
    "035.Purple_Finch": "Purple Finch",
    "042.Vermilion_Flycatcher": "Vermilion Flycatcher",
    "048.European_Goldfinch": "European Goldfinch",
    "050.Eared_Grebe": "Eared Grebe",
    "059.California_Gull": "California Gull",
    "068.Ruby_throated_Hummingbird": "Ruby-throated Hummingbird",
    "073.Blue_Jay": "Blue Jay",
    "081.Pied_Kingfisher": "Pied Kingfisher",
    "095.Baltimore_Oriole": "Baltimore Oriole",
    "101.White_Pelican": "White Pelican",
    "106.Horned_Puffin": "Horned Puffin",
    "108.White_necked_Raven": "White-necked Raven",
    "112.Great_Grey_Shrike": "Great Grey Shrike",
    "118.House_Sparrow": "House Sparrow",
    "134.Cape_Glossy_Starling": "Cape Glossy Starling",
    "138.Tree_Swallow": "Tree Swallow",
    "144.Common_Tern": "Common Tern",
    "191.Red_headed_Woodpecker": "Red-headed Woodpecker"
}

class DialogueSystem:
    def __init__(self, test_mode=False):
        # Initialize with optional test mode (no speech recognition)
        self.test_mode = test_mode
        
        # Initialize speech recognizer if not in test mode
        if not test_mode:
            try:
                self.recognizer = sr.Recognizer()
                # Test microphone access
                with sr.Microphone() as source:
                    pass
                print("Speech recognition initialized successfully")
            except Exception as e:
                print(f"Error initializing speech recognition: {e}")
                print("Falling back to test mode...")
                self.test_mode = True
        
        # Initialize gender recognition model
        print("Loading gender recognition model...")
        model_name = "dima806/fairface_gender_image_detection"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.model.eval()
        self.labels = self.model.config.id2label if hasattr(self.model.config, "id2label") else {0: "Male", 1: "Female"}
        print("Model loaded successfully")

        # Import rclpy here to access the latest detected_rings information
        # Note: We'll actually get this from RobotCommander
        self.detected_rings = {}
        self.detected_birds = {}
        
        # Initialize pygame for audio playback
        try:
            pygame.mixer.init()
            print("Audio playback initialized successfully")
        except Exception as e:
            print(f"Error initializing audio playback: {e}")
            self.test_mode = True
        
    def set_detected_birds_and_rings(self, detected_birds, detected_rings):
        """Set the detected birds and rings from RobotCommander"""
        self.detected_birds = detected_birds
        self.detected_rings = detected_rings
        
    def get_bird_location_description(self, bird_id):
        """Generate a location description based on the associated ring"""
        if not self.detected_birds or not self.detected_rings or bird_id not in self.detected_birds:
            return "somewhere in the area"
        
        bird_data = self.detected_birds[bird_id]
        ring_id = bird_data.get('associated_ring')
        
        if not ring_id or ring_id not in self.detected_rings:
            return "somewhere in the area"
        
        ring_data = self.detected_rings[ring_id]
        ring_color = ring_data.get('color', 'unknown')
        ring_geo = ring_data.get('geo', 'unknown')
        
        return f"in the {ring_geo} part of the park near the {ring_color} ring"
        
    def detect_gender(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_id = probs.argmax()
                pred_label = self.labels[pred_id]
                confidence = probs[pred_id]
            return pred_label, confidence
        except Exception as e:
            print(f"Error in gender detection: {e}")
            return "Unknown", 0.0
        
    def speak(self, text):
        """Convert text to speech, play it, and update the conversation window."""
        print(f"Robot: {text}")
        update_conversation_window("Conversation", f"Robot: {text}")  # Update the window with new text
        
        if self.test_mode:
            return
        
        try:
            # Generate speech file
            tts = gTTS(text=text, lang='en')
            tts.save("temp_speech.mp3")
            
            # Play the speech
            pygame.mixer.music.load("temp_speech.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
            # Clean up the temporary file
            time.sleep(0.5)
            if os.path.exists("temp_speech.mp3"):
                os.remove("temp_speech.mp3")
        except Exception as e:
            print(f"Error in speech synthesis: {e}")
            
    def listen(self, timeout=10, phrase_time_limit=60, gender="Person"):
        """Listen for user speech, convert to text, and update the conversation window."""
        if self.test_mode:
            response = input(f"{gender} (type response): ")
            print(f"{gender}: {response}")
            update_conversation_window("Conversation", f"{gender}: {response}")  # Update the window with new text
            return response.lower()
        
        try:
            with sr.Microphone() as source:
                print(f"Listening... (timeout: {timeout}s, phrase limit: {phrase_time_limit}s)")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        
            text = self.recognizer.recognize_google(audio)
            print(f"{gender}: {text}")
            update_conversation_window("Conversation", f"{gender}: {text}")  # Update the window with new text
            return text.lower()
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return None
            
    def extract_bird_name(self, response):
        """Extract the bird name from a response."""
        if not response:
            return None
    
        response = response.lower()
        
        # First try exact matches with bird IDs
        for bird in COMMON_BIRDS:
            bird_clean = bird.lower()
            if bird_clean in response:
                return bird
        
        # If no exact match, try with display names
        for bird_id, display_name in BIRD_DISPLAY_NAMES.items():
            if display_name.lower() in response:
                return bird_id
                
        # Try with partial matches (just the actual bird name)
        for bird in COMMON_BIRDS:
            bird_parts = bird.split('.')[-1].lower().replace('_', ' ')
            if bird_parts in response:
                return bird
                
        return None

    def find_bird_by_name(self, bird_name):
        """Find a bird ID based on name from detected birds"""
        if not self.detected_birds:
            return None
            
        # Clean input name
        bird_name = bird_name.lower().replace('_', ' ')
        
        for bird_id, bird_data in self.detected_birds.items():
            detected_name = bird_data.get('name', '').lower()
            if bird_name in detected_name or detected_name in bird_name:
                return bird_id
                
        return None
            
    def talk_to_female(self, gender="Female"):
        """Dialogue with a female person."""
        reset_conversation_window("Conversation")  # Reset the window for a new conversation
        self.speak("Hey girlie, which is your favorite bird?")
        
        # Keep asking until we get a valid bird name
        while True:
            bird_response = self.listen(gender=gender)
            if not bird_response:
                self.speak("I couldn't get that, could you repeat your favorite bird?")
                continue

            bird_name = self.extract_bird_name(bird_response)
            if not bird_name:
                self.speak("I couldn't get that, could you repeat your favorite bird?")
                continue
                
            # If we get here, we have a valid bird name
            break
        
        # Try to find this bird in our detected birds    
        bird_id = self.find_bird_by_name(bird_name)
        
        if bird_id and bird_id in self.detected_birds:
            # We have this bird in our detected birds
            bird_display_name = BIRD_DISPLAY_NAMES.get(bird_name, bird_name)
            location = self.get_bird_location_description(bird_id)
            self.speak(f"Well, there is one {bird_display_name} {location}.")
        else:
            # We don't have this bird in detected birds
            bird_display_name = BIRD_DISPLAY_NAMES.get(bird_name, bird_name)
            self.speak(f"Well, I haven't seen any {bird_display_name} around here yet.")
            
        return bird_name

            
    def talk_to_male(self, gender="Male"):
        """Dialogue with a male person with error handling for invalid and negative responses."""
        reset_conversation_window("Conversation")  # Reset the window for a new conversation

        def confirm_and_respond(bird):
            bird_id = self.find_bird_by_name(bird)
            bird_display_name = BIRD_DISPLAY_NAMES.get(bird, bird)
            
            if bird_id and bird_id in self.detected_birds:
                location = self.get_bird_location_description(bird_id)
                self.speak(f"OK. The {bird_display_name} then. There is one {location}.")
            else:
                self.speak(f"OK. The {bird_display_name} then. I haven't seen any around here yet.")


        bird_counts = {}  # Track bird occurrences
        self.speak("Hey broski, which is your favourite bird?")
        last_bird = None

        while True:
            response = self.listen(gender=gender)
            if not response:
                prompt = f"I couldn't get that, could you repeat your favorite bird?" if not last_bird else f"I couldn't get that, are you sure your favorite bird is {BIRD_DISPLAY_NAMES.get(last_bird, last_bird)}?"
                self.speak(prompt)
                continue

            current_bird = self.extract_bird_name(response)
            if not current_bird:
                prompt = f"I couldn't get that, could you repeat your favorite bird?" if not last_bird else f"I couldn't get that, are you sure your favorite bird is {BIRD_DISPLAY_NAMES.get(last_bird, last_bird)}?"
                self.speak(prompt)
                continue

            # Update bird counts
            bird_counts[current_bird] = bird_counts.get(current_bird, 0) + 1
            last_bird = current_bird

            # Check if bird has been mentioned twice
            if bird_counts[current_bird] >= 2:
                confirm_and_respond(current_bird)
                return current_bird

            self.speak("Are you sure?")
            confirmation = self.listen()
            if not confirmation:
                self.speak(f"I couldn't get that, are you sure your favorite bird is {BIRD_DISPLAY_NAMES.get(current_bird, current_bird)}?")
                continue

            # Check for negative responses
            if any(neg in confirmation for neg in ["no", "not", "not really", "nah"]):
                self.speak("Which is your favorite bird?")
                continue

            if any(aff in confirmation for aff in ["yes", "sure", "yeah"]):
                confirm_and_respond(current_bird)
                return current_bird

            new_bird = self.extract_bird_name(confirmation)
            if not new_bird:
                self.speak(f"I couldn't get that, are you sure your favorite bird is {BIRD_DISPLAY_NAMES.get(current_bird, current_bird)}?")
                continue

            # Update bird counts for new bird
            bird_counts[new_bird] = bird_counts.get(new_bird, 0) + 1
            last_bird = new_bird

            # Check if new bird has been mentioned twice
            if bird_counts[new_bird] >= 2:
                confirm_and_respond(new_bird)
                return new_bird

            self.speak(f"OK, the {BIRD_DISPLAY_NAMES.get(new_bird, new_bird)} then. Are you sure?")
            
    def run_dialogue(self, image_path):
        """Run the full dialogue system with gender recognition."""
        print(f"Processing image: {image_path}")
        
        # Detect gender
        gender, conf = self.detect_gender(image_path)
        print(f"Detected gender: {gender} with confidence {conf:.4f}")
        
        # Start appropriate dialogue based on gender
        if gender == "Female":
            favorite_bird = self.talk_to_female()
        else:  # Male or Unknown defaults to male dialogue pattern
            favorite_bird = self.talk_to_male()
            
        print(f"Favorite bird determined: {favorite_bird}")
        return favorite_bird

conversation_history = ""  # Global variable to store the conversation history
window_created = False  # Global flag to track if the window has been created

def update_conversation_window(window_name, new_text, width=1000, height=400, font_scale=0.6, font_color=(255, 255, 255)):
    """Update the conversation window with new text."""
    global conversation_history, window_created
    conversation_history += f"{new_text}\n"  # Append new text with a newline

    # Create a blank image (black canvas)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Split the conversation history into lines
    lines = conversation_history.split('\n')  # Split by newline for proper rendering

    # Render each line of text
    y_offset = 30  # Initial Y offset for the text
    line_height = 25  # Space between lines
    for line in lines:
        if line.strip():  # Skip empty lines
            cv2.putText(
                canvas,
                line,
                (10, y_offset),  # X and Y position
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_color,
                1,
                cv2.LINE_AA
            )
            y_offset += line_height
            if y_offset > height - 10:  # Stop rendering if text exceeds the canvas height
                break

    # Display the canvas in a window
    cv2.imshow(window_name, canvas)
    cv2.waitKey(1)  # Refresh the window
    window_created = True  # Set the flag to indicate the window has been created

def reset_conversation_window(window_name):
    """Reset the conversation window by clearing the history and closing the window."""
    global conversation_history, window_created
    conversation_history = ""  # Clear the conversation history
    
    if window_created:  # Only destroy the window if it has been created
        cv2.destroyWindow(window_name)
        window_created = False  # Reset the flag


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bird preference dialogue system')
    parser.add_argument('--image', type=str, default="faces/person3_F.png", 
                        help='Path to the image to analyze')
    parser.add_argument('--test', action='store_true', 
                        help='Run in test mode (no speech recognition, text input instead)')
    
    args = parser.parse_args()
    
    dialogue_system = DialogueSystem(test_mode=args.test)
    dialogue_system.run_dialogue(args.image)