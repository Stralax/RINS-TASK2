import os
import torch
from PIL import Image
import speech_recognition as sr
from gtts import gTTS
import pygame
import time
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import argparse

COMMON_BIRDS = [
    "robin", "hawk", "eagle", "swallow", "sparrow", "stork", 
    "cuckoo", "penguin", "duck", "goose", "owl", "pigeon", "crow",
    "raven", "seagull", "hummingbird", "flamingo", "peacock", "parrot",
    "laysan albatross", "yellow headed blackbird", "indigo bunting",
    "pelagic cormorant", "american crow", "yellow billed cuckoo",
    "purple finch", "vermilion flycatcher", "european goldfinch",
    "eared grebe", "california gull", "ruby throated hummingbird",
    "blue jay", "pied kingfisher", "baltimore oriole",
    "white pelican", "horned puffin", "white necked raven",
    "great grey shrike", "house sparrow", "cape glossy starling",
    "tree swallow", "common tern", "red headed woodpecker"
]

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

        # Bird locations (for the final response)
        self.bird_locations = {
            "robin": "center of the park sitting on a green ring",
            "hawk": "central part of the park sitting on a yellow ring",
            "swallow": "west part of the park sitting on a red ring",
            "eagle": "south part of the park sitting on a white ring",
            "sparrow": "north part of the park sitting on a brown ring",
            "stork": "southwest part of the park sitting on a black ring",
            "cuckoo": "east part of the park sitting on a blue ring"
        }
        
        # Initialize pygame for audio playback
        try:
            pygame.mixer.init()
            print("Audio playback initialized successfully")
        except Exception as e:
            print(f"Error initializing audio playback: {e}")
            self.test_mode = True
        
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
        """Convert text to speech and play it."""
        print(f"Robot: {text}")
        
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
            time.sleep(0.5)  # Small delay to ensure file is not in use
            if os.path.exists("temp_speech.mp3"):
                os.remove("temp_speech.mp3")
        except Exception as e:
            print(f"Error in speech synthesis: {e}")
            
    def listen(self, timeout=10, phrase_time_limit=40):
        """Listen for user speech and convert to text.
        
        Args:
            timeout (int): Maximum number of seconds to wait for speech to start
            phrase_time_limit (int): Maximum number of seconds to allow a phrase to continue
        
        Returns:
            str or None: Recognized text (lowercase) or None if recognition failed
        """
        if self.test_mode:
            # In test mode, simulate responses
            response = input("Person (type response): ")
            print(f"Person: {response}")
            return response.lower()
            
        try:
            with sr.Microphone() as source:
                print(f"Listening... (timeout: {timeout}s, phrase limit: {phrase_time_limit}s)")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                
            text = self.recognizer.recognize_google(audio)
            print(f"Person: {text}")
            return text.lower()
        except sr.WaitTimeoutError:
            print("Listening timed out - no speech detected")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service")
            return None
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return None
            
    def extract_bird_name(self, response):
        """Extract the bird name from a response."""
        if not response:
            return None
    
        response = response.lower()
        
        # Check if any bird name is in the response
        for bird in COMMON_BIRDS:
            if bird in response:
                return bird
                
        return None
            
    def talk_to_female(self):
        """Dialogue with a female person."""
        self.speak("Hey girlie, which is your favorite bird?")
        
        # Keep asking until we get a valid bird name
        while True:
            bird_response = self.listen()
            if not bird_response:
                self.speak("I couldn't get that, could you repeat your favorite bird?")
                continue

            bird_name = self.extract_bird_name(bird_response)
            if not bird_name:
                self.speak("I couldn't get that, could you repeat your favorite bird?")
                continue
                
            # If we get here, we have a valid bird name
            break
            
        location = self.bird_locations.get(bird_name, "park")
        self.speak(f"Well, there is one {bird_name} in the {location}.")
        return bird_name

            
    def talk_to_male(self):
        """Dialogue with a male person with error handling for invalid and negative responses."""
        def confirm_and_respond(bird):
            location = self.bird_locations.get(bird, "somewhere in the park")
            self.speak(f"OK. The {bird} then. There is one in the {location}.")

        bird_counts = {}  # Track bird occurrences
        self.speak("Hey broski, which is your favourite bird?")
        last_bird = None

        while True:
            response = self.listen()
            if not response:
                prompt = f"I couldn't get that, could you repeat your favorite bird?" if not last_bird else f"I couldn't get that, are you sure your favorite bird is {last_bird}?"
                self.speak(prompt)
                continue

            current_bird = self.extract_bird_name(response)
            if not current_bird:
                prompt = f"I couldn't get that, could you repeat your favorite bird?" if not last_bird else f"I couldn't get that, are you sure your favorite bird is {last_bird}?"
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
                self.speak(f"I couldn't get that, are you sure your favorite bird is {current_bird}?")
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
                self.speak(f"I couldn't get that, are you sure your favorite bird is {current_bird}?")
                continue

            # Update bird counts for new bird
            bird_counts[new_bird] = bird_counts.get(new_bird, 0) + 1
            last_bird = new_bird

            # Check if new bird has been mentioned twice
            if bird_counts[new_bird] >= 2:
                confirm_and_respond(new_bird)
                return new_bird

            self.speak(f"OK, the {new_bird} then. Are you sure?")
            
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bird preference dialogue system')
    parser.add_argument('--image', type=str, default="faces/person3_F.png", 
                        help='Path to the image to analyze')
    parser.add_argument('--test', action='store_true', 
                        help='Run in test mode (no speech recognition, text input instead)')
    
    args = parser.parse_args()
    
    dialogue_system = DialogueSystem(test_mode=args.test)
    dialogue_system.run_dialogue(args.image)