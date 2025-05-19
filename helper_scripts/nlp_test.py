import os
import torch
from PIL import Image
import speech_recognition as sr
from gtts import gTTS
import pygame
import time
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import argparse

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
            
    def listen(self):
        """Listen for user speech and convert to text."""
        if self.test_mode:
            # In test mode, simulate responses
            response = input("Person (type response): ")
            print(f"Person: {response}")
            return response.lower()
            
        try:
            with sr.Microphone() as source:
                print("Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source)
                
            text = self.recognizer.recognize_google(audio)
            print(f"Person: {text}")
            return text.lower()
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
            return "unknown"
            
        # List of common bird names to check for
        common_birds = ["robin", "hawk", "eagle", "swallow", "sparrow", "stork", 
                       "cuckoo", "penguin", "duck", "goose", "owl", "pigeon", "crow",
                       "raven", "seagull", "hummingbird", "flamingo", "peacock", "parrot"]
        
        response = response.lower()
        
        # Check if any bird name is in the response
        for bird in common_birds:
            if bird in response:
                return bird
                
        # If no specific bird found, return the entire response
        return response
            
    def talk_to_female(self):
        """Dialogue with a female person."""
        self.speak("Hi woman, which is your favorite bird?")
        
        bird_response = self.listen()
        if bird_response:
            bird_name = self.extract_bird_name(bird_response)
            location = self.bird_locations.get(bird_name, "somewhere in the park")
            self.speak(f"Well there is one {bird_name} in the {location}.")
            return bird_name
        else:
            self.speak("I'm sorry, I didn't catch that.")
            return None
            
    def talk_to_male(self):
        """Dialogue with a male person."""
        self.speak("Hi man, which is your favorite bird?")
        
        bird_choices = []
        asked_count = 0
        
        while True:
            # First time or after rejection, listen for bird preference
            bird_response = self.listen()
            if not bird_response:
                self.speak("I'm sorry, I didn't catch that.")
                continue
                
            bird_name = self.extract_bird_name(bird_response)
            bird_choices.append(bird_name)
            
            # Always ask if they're sure
            self.speak("Are you sure?")
            confirmation = self.listen()
            
            # Process confirmation response
            if not confirmation:
                self.speak("I'm sorry, I didn't catch that. Let's try again.")
                continue
                
            # Case 1: Direct confirmation ("Yes, I'm sure")
            if "yes" in confirmation.lower() or "sure" in confirmation.lower() or "yeah" in confirmation.lower():
                location = self.bird_locations.get(bird_name, "somewhere in the park")
                self.speak(f"There is one {bird_name} in the {location}.")
                return bird_name
                
            # Case 2: New bird mentioned in uncertainty response
            new_bird_name = self.extract_bird_name(confirmation)
            if new_bird_name != "sure" and new_bird_name in self.bird_locations:
                bird_choices.append(new_bird_name)
                self.speak(f"OK, the {new_bird_name} then. Are you sure?")
                
                second_confirmation = self.listen()
                if not second_confirmation:
                    self.speak("I didn't catch that. Let me ask again.")
                    continue
                    
                # Handle the response to the follow-up question
                if "yes" in second_confirmation.lower() or "sure" in second_confirmation.lower():
                    location = self.bird_locations.get(new_bird_name, "somewhere in the park")
                    self.speak(f"OK. The {new_bird_name} then. There is one in the {location}.")
                    return new_bird_name
                
                # They're still uncertain, extract any new bird mentioned
                another_bird = self.extract_bird_name(second_confirmation)
                if another_bird != "sure" and another_bird in self.bird_locations:
                    bird_choices.append(another_bird)
                    asked_count += 1
                    
                    # For third or more time, be more direct
                    if asked_count >= 2:
                        self.speak(f"Are you sure now about the {another_bird}?")
                    else:
                        self.speak(f"OK, the {another_bird} then. Are you sure?")
                    
                    final_confirmation = self.listen()
                    if final_confirmation and ("yes" in final_confirmation.lower() or "sure" in final_confirmation.lower()):
                        location = self.bird_locations.get(another_bird, "somewhere in the park")
                        self.speak(f"OK. The {another_bird} then. There is one in the {location}.")
                        return another_bird
            
            # Check if any bird has been mentioned twice (their actual preference)
            for bird in bird_choices:
                if bird_choices.count(bird) >= 2:
                    location = self.bird_locations.get(bird, "somewhere in the park")
                    self.speak(f"OK. The {bird} then. There is one in the {location}.")
                    return bird
            
            # If we've been through several rounds and still no clear choice
            asked_count += 1
            if asked_count >= 3:
                self.speak("OK, let me ask one final time. Which bird is truly your favorite?")
                final_response = self.listen()
                if final_response:
                    final_bird = self.extract_bird_name(final_response)
                    location = self.bird_locations.get(final_bird, "somewhere in the park")
                    self.speak(f"Great! There is one {final_bird} in the {location}.")
                    return final_bird
                else:
                    # If all else fails, pick the most frequently mentioned bird
                    most_common = max(set(bird_choices), key=bird_choices.count)
                    location = self.bird_locations.get(most_common, "somewhere in the park")
                    self.speak(f"I'll go with the {most_common} then. There is one in the {location}.")
                    return most_common
            
            # Continue the conversation more naturally
            phrases = [
                "OK, which bird do you prefer then?", 
                "So which bird would you say is your actual favorite?",
                "Let's try once more. What's your favorite bird?"
            ]
            self.speak(phrases[asked_count % len(phrases)])
                  
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