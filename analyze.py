from google import genai
from google.genai import types
from PIL import Image
import io
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Secure credential handling - load from .env file
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please update your .env file.")

client = genai.Client(api_key=gemini_api_key)

def get_llm_response(image_data: bytes) -> str:
    image = Image.open(io.BytesIO(image_data))
    
    # Converting image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Prompt for identifying what mood an inidvidual is in from their facial expressions
    prompt = """
    Analyze the mood of the person in this image using their facial expressions. Do this by identifying visible emotions, key facial features (eyes, mouth, eyebrows), overall expressions, and the dominant or mixed emotional atmosphere in relation to the context. Return the response in a single sentence.
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=prompt),
                        types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=img_byte_arr))
                    ]
                )
            ]
        )
        
        if response.text:
            return response.text
        else:
            return "Facial expression analysis failed. Please try again."
            
    except Exception as e:
        return f"Error analyzing image: {str(e)}"