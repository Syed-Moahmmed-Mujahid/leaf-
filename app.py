import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
from io import BytesIO

# --- CRITICAL CONFIGURATION: GEMINI API KEY ---

# The genai.Client() and genai.configure() require the API key.
# We securely load it from Streamlit's secrets management first.
try:
    # Attempt to load the key from .streamlit/secrets.toml (preferred) or OS environment
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY"))
    
    if not GEMINI_API_KEY:
        st.error("Configuration Error: GEMINI_API_KEY not found. Please set it in .streamlit/secrets.toml.")
        # Stops the application if the key is missing
        st.stop()
        
    genai.configure(api_key=GEMINI_API_KEY)
    
except Exception as e:
    st.error(f"Configuration Error: Could not configure the Gemini API. Details: {e}")
    st.stop()


def get_gemini_response(input_prompt, image_parts):
    """
    Sends the user's prompt and image data to the Gemini 2.5 Flash model for detailed analysis.
    """
    # Use the generative model client
    model = genai.GenerativeModel('gemini-2.5-flash')
    # The image_parts list contains the image data. We combine the text prompt and the first image part.
    response = model.generate_content([input_prompt, image_parts[0]])
    return response.text

def input_image_setup(uploaded_file, captured_image_bytes):
    """
    Prepare the uploaded file or captured image bytes for the Gemini API call.
    Returns a list of image parts in the required format for the API.
    """
    bytes_data = None
    mime_type = None
    
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        mime_type = uploaded_file.type
    elif captured_image_bytes is not None:
        bytes_data = captured_image_bytes
        # Assume camera input returns a JPEG image
        mime_type = "image/jpeg"
    else:
        # Should not happen if the submit button is properly guarded
        raise FileNotFoundError("No file uploaded or image captured.")
    
    image_parts = [
        {
            "mime_type": mime_type,
            "data": bytes_data
        }
    ]
    return image_parts

# --- Streamlit UI Setup ---
st.set_page_config(page_title="AI Plant Disease Detector", layout="wide")
st.header("AI Plant Disease Detector 🌿🔬")
st.markdown("Upload an image of a leaf to get an expert diagnosis, symptom analysis, and treatment plan.")

# --- Image Input Section ---
st.subheader("Leaf Image Input")
source = st.radio("Choose image source:", ("Upload an image", "Take a picture"))

uploaded_file = None
captured_image_bytes = None

# Use columns for better layout of input and display
col1, col2 = st.columns(2)

with col1:
    if source == "Upload an image":
        uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    else:  # Take a picture
        captured_image = st.camera_input("Take a picture of the plant leaf")
        if captured_image:
            captured_image_bytes = captured_image.getvalue()

# Display image in the second column
with col2:
    display_image = None
    caption = ""
    
    if uploaded_file is not None:
        try:
            display_image = Image.open(uploaded_file)
            caption = "Uploaded Leaf Image."
        except Exception as e:
            st.error(f"Error loading image: {e}")
            
    elif captured_image_bytes is not None:
        # Re-open bytes as PIL Image to ensure correct display
        display_image = Image.open(BytesIO(captured_image_bytes))
        caption = "Captured Leaf Image."

    if display_image:
        st.image(display_image, caption=caption, use_column_width=True)

st.divider()
submit = st.button("Analyze Leaf Disease")

# --- Processing Logic ---
if submit:
    if uploaded_file is None and captured_image_bytes is None:
        st.warning("Please upload a leaf image or take a picture to proceed with the analysis.")
    else:
        try:
            # Prepare image data for the API call
            image_data = input_image_setup(uploaded_file, captured_image_bytes)

            # Customized prompt for plant pathology analysis
            input_prompt = """
            You are an expert agricultural scientist and certified plant pathologist. Your task is to analyze the provided image of a plant leaf and give a complete, detailed diagnosis and recommendation.

            Structure your response into the following three sections using markdown headings:

            ### 1. Plant & Disease Identification
            - **Plant Species (if identifiable):** [Identify the specific plant/crop]
            - **Diagnosis:** [Identify the most probable disease, pest, or deficiency (e.g., Late Blight, Spider Mites, Iron Deficiency)]
            - **Cause Type:** [e.g., Fungal, Bacterial, Viral, Insect Pest, Nutrient Deficiency, Environmental Stress]

            ### 2. Symptom Analysis
            - Describe the specific visual symptoms observed in the image (e.g., presence of chlorosis, necrosis, lesions, mold, wilting pattern).
            - Explain what these symptoms indicate about the health of the plant and the progression of the issue.

            ### 3. Recommended Treatment & Prevention
            - **Immediate Treatment:** Provide specific, actionable steps to treat the identified issue (e.g., suggested fungicide/pesticide type, proper pruning, isolation).
            - **Long-term Prevention:** Suggest strategies for preventing recurrence, including optimal soil management, watering schedules, and proper environmental conditions.

            Maintain a professional, informative, and easy-to-understand tone suitable for a gardener or farmer.
            """

            # Call the Gemini API with the specialized prompt and image data
            with st.spinner("Analyzing the leaf image and generating the plant health report... 🧪🌱"):
                response = get_gemini_response(input_prompt, image_data)

            st.subheader("🔬 Plant Health Report and Treatment Plan")
            # Display the detailed, structured response
            st.markdown(response)

        except FileNotFoundError as fnfe:
            st.error(f"File Error: {fnfe}")

        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {e}")
            print(f"Full Error Trace: {e}")