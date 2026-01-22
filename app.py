import datetime
import hashlib
import os
import json
import base64
import time
import asyncio
import requests
import io
import textwrap
import streamlit as st
import edge_tts
from PIL import Image
import random

# ---------------- IMPORTS ----------------

from dotenv import load_dotenv
from openai import OpenAI

# ---------- 1. CRITICAL LIBRARY FIXES ----------
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

try:
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
except ImportError:
    from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

# ---------------- ENV ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QUBRID_API_KEY = os.getenv("QUBRID_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")

if not OPENAI_API_KEY or not STABILITY_API_KEY:
    st.error("⚠️ Missing OPENAI_API_KEY or STABILITY_API_KEY in .env")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- UI Styling ----------------
st.set_page_config(page_title="⚡ Mythos AI Studio", page_icon="🕉️", layout="wide")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Eczar:wght@400;700&display=swap');
    
    body { background: radial-gradient(circle at top, #120f0c, #050402); color: #ffedd5; }
    h1 { font-family: 'Cinzel', serif; color: #facc15; text-shadow: 0 0 12px rgba(250, 204, 21, 0.6); }
    
    .stButton>button { 
        background: linear-gradient(135deg, #92400e, #facc15); 
        color: black; 
        font-family: 'Cinzel', serif;
        font-weight: bold;
        border-radius: 14px; 
    }
    
    .stTextInput>div>input { 
        background:#1a1a1a; 
        color:#ffedd5; 
        border:1px solid #d97706; 
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Mythos AI Studio (Qwen Edition)")
st.caption("Mythological video generator using AI")

# ---------------- UTIL ----------------

CAMERAS = [
    "wide cinematic shot",
    "medium three-quarter shot",
    "low angle heroic shot",
    "close-up divine portrait"
]

QUBRID_URL = "https://platform.qubrid.com/api/v1/qubridai/image/generation"

# ---------------- 2. CHARACTER MAPPING ----------------
CHARACTER_MAP = {
    "shiva": "characters/Shiva.png",
    "mahadev": "characters/Shiva.png",
    "shankar": "characters/Shiva.png",
    
    "hanuman": "characters/Hanuman.png",
    "bajrang": "characters/Hanuman.png",
    
    "krishna": "characters/Krishna.png",
    "kanha": "characters/Krishna.png",
    
    "rama": "characters/Rama.png",
    "ram": "characters/Rama.png",
    "raghu": "characters/Rama.png",
    
    "arjun": "characters/Lord Arjuna.png",
    "arjuna": "characters/Lord Arjuna.png"
}

# ---------------- Helpers ----------------
def detect_language_simple(text: str) -> str:
    if not text: return "en"
    for ch in text:
        if '\u0900' <= ch <= '\u097F':
            return "hi"
    return "en"

def choose_voice_for_text(text: str) -> str:
    if detect_language_simple(text) == "hi":
        return "hi-IN-MadhurNeural"
    return "en-IN-PrabhatNeural"

def get_character_reference_from_topic(text: str) -> str | None:
    t = text.lower()
    
    # SMART CHECK: If prompt implies multiple people ("Ram AND Sita"), 
    # we should likely SKIP single-character reference to allow group generation.
    if " and " in t or " with " in t:
        return None  # Force Text-to-Image for group shots
        
    for key, filepath in CHARACTER_MAP.items():
        if key in t:
            if os.path.exists(filepath):
                return filepath
    return None

# ---------------- 3. SUBTITLE GENERATION (PIL) ----------------
def add_subtitles_to_image(img_path: str, text: str):
    try:
        with PIL.Image.open(img_path) as img:
            img = img.convert("RGB")
            draw = PIL.ImageDraw.Draw(img)
            w, h = img.size
            
            fontsize = 40
            try:
                font = PIL.ImageFont.truetype("Arial Unicode.ttf", fontsize)
            except IOError:
                try:
                    font = PIL.ImageFont.truetype("arial.ttf", fontsize)
                except IOError:
                    font = PIL.ImageFont.load_default()

            lines = textwrap.wrap(text, width=50)
            line_height = fontsize + 10
            text_box_height = len(lines) * line_height + 40
            
            overlay = PIL.Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = PIL.ImageDraw.Draw(overlay)
            
            rect_y0 = h - text_box_height - 20
            rect_y1 = h - 20
            
            overlay_draw.rectangle(
                [(20, rect_y0), (w - 20, rect_y1)], 
                fill=(0, 0, 0, 180)
            )
            
            img = PIL.Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
            draw = PIL.ImageDraw.Draw(img)
            
            current_y = rect_y0 + 20
            for line in lines:
                try:
                    left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
                    text_w = right - left
                except AttributeError:
                    text_w, _ = draw.textsize(line, font=font)
                
                text_x = (w - text_w) / 2
                draw.text((text_x, current_y), line, font=font, fill="white")
                current_y += line_height
            
            output_path = img_path.replace(".png", "_sub.png")
            img.save(output_path)
            return output_path

    except Exception as e:
        print(f"Subtitle Error: {e}")
        return img_path 

# ---------------- 4. AI LOGIC ----------------

def build_cinematic_script_prompt(topic: str, lang: str) -> str:
    language_instruction = "Narrate ONLY in Hindi (Devanagari)." if lang == "hi" else "Narrate ONLY in English."
    return f"""
You are a legendary mythological filmmaker. Write a VISUALLY STUNNING 4-scene script for a short cinematic video.

Topic: {topic}

CRITICAL REQUIREMENTS:
1. You MUST create EXACTLY 4 scenes - no more, no less
2. Each scene must be substantially different visually
3. Output must be valid JSON with this EXACT structure:

{{
  "scenes": [
    {{
      "narration": "Scene 1 narration here",
      "image_prompt": "Detailed visual description for scene 1"
    }},
    {{
      "narration": "Scene 2 narration here",
      "image_prompt": "Detailed visual description for scene 2"
    }},
    {{
      "narration": "Scene 3 narration here",
      "image_prompt": "Detailed visual description for scene 3"
    }},
    {{
      "narration": "Scene 4 narration here",
      "image_prompt": "Detailed visual description for scene 4"
    }}
  ]
}}

Rules:
- "narration": {language_instruction} (Max 2 short sentences per scene)
- "image_prompt": Highly detailed English description of the visual scene, action, characters, and environment
- Each scene should show progression in the story
- Make each image_prompt DISTINCT and DIFFERENT from the others

Example image_prompt: "Lord Rama standing in a lush forest, golden hour lighting, divine glow around him, ancient trees in background, wearing orange robes, bow in hand, peaceful expression"
"""

def generate_script_gpt4(topic: str):
    lang = detect_language_simple(topic)
    prompt = build_cinematic_script_prompt(topic, lang)

    try:
        res = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert mythological film director. You always output valid JSON with exactly 4 scenes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        txt = res.choices[0].message.content
        clean_txt = txt.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_txt)
        
        # Validate script structure
        if "scenes" in data:
            scenes = data["scenes"]
        elif isinstance(data, list):
            scenes = data
        else:
            # Try to extract scenes from any key
            scenes = list(data.values())[0] if data else []
        
        # CRITICAL: Ensure exactly 4 scenes
        if not isinstance(scenes, list) or len(scenes) != 4:
            st.warning(f"⚠️ Script returned {len(scenes) if isinstance(scenes, list) else 0} scenes instead of 4. Regenerating...")
            return None
            
        # Validate each scene has required fields
        for i, scene in enumerate(scenes):
            if not isinstance(scene, dict):
                st.error(f"Scene {i+1} is not a valid object")
                return None
            if "narration" not in scene or "image_prompt" not in scene:
                st.error(f"Scene {i+1} missing required fields")
                return None
                
        return scenes

    except json.JSONDecodeError as e:
        st.error(f"Script JSON Parse Error: {e}")
        return None
    except Exception as e:
        st.error(f"Script Gen Error: {e}")
        return None

# ---------- 5. IMAGE GENERATION (Stable Diffusion 3.5 Large) ----------

def generate_scene_image_sd(scene_prompt, ref_path, index):
    """
    Generates an image using Stable Diffusion 3.5 Large via Qubrid.
    
    🔧 FIXES:
    - Uses dynamic seed based on index + random component
    - Implements proper retry logic with exponential backoff
    - Better error handling and logging
    """
    
    if index is None:
        index = 0

    camera = CAMERAS[index % len(CAMERAS)]
    
    # Try to extract character name from ref_path if available to help consistency
    char_name = ""
    if ref_path:
        char_name = os.path.basename(ref_path).split('.')[0]

    final_prompt = f"""
{char_name if char_name else ''} {scene_prompt}, 
{camera}, 
ancient Indian mythological environment, 
divine glow, ultra detailed, cinematic lighting, 8k, 
no modern objects, no text, no watermark
""".strip()

    # 🔧 FIX: Dynamic seed generation
    # Use index + random component to ensure each scene is different
    base_seed = int(time.time()) % 10000
    scene_seed = base_seed + (index * 1000) + random.randint(0, 999)

    try:
        headers = {
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "stabilityai/stable-diffusion-3.5-large",
            "positive_prompt": final_prompt,
            "width": 1024,
            "height": 1024,
            "steps": 30,
            "cfg": 7.5,
            "seed": scene_seed,  # 🔧 FIX: Dynamic seed instead of fixed 50
            "negative_prompt": "modern buildings, text, watermark, blur, low quality, deformed, extra limbs, animals, sheep, lamb"
        }

        # 🔧 FIX: Improved retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                st.caption(f"🎨 Generating image {index+1} (attempt {attempt+1}/{max_retries}, seed: {scene_seed})...")
                
                r = requests.post(QUBRID_URL, headers=headers, json=data, timeout=120)
                
                if r.status_code == 200:
                    filename = f"scene_{index}_{scene_seed}.png"
                    with open(filename, "wb") as f:
                        f.write(r.content)
                    
                    # Verify image was created and is valid
                    if os.path.exists(filename) and os.path.getsize(filename) > 1000:
                        st.caption(f"✅ Scene {index+1} generated successfully")
                        return filename
                    else:
                        st.warning(f"⚠️ Image file seems invalid, retrying...")
                        
                elif r.status_code == 429:
                    wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                    st.warning(f"⏳ Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    st.error(f"❌ API Error {r.status_code}: {r.text[:200]}")
                    time.sleep(3 * (attempt + 1))
                    
            except requests.Timeout:
                st.warning(f"⏳ Request timed out on attempt {attempt+1}")
                time.sleep(5 * (attempt + 1))
            except Exception as e:
                st.error(f"❌ Exception on attempt {attempt+1}: {str(e)[:100]}")
                time.sleep(3 * (attempt + 1))

        st.error(f"❌ Failed to generate scene {index+1} after {max_retries} attempts")
        return None
                
    except Exception as e:
        st.error(f"SD Generation Exception: {e}")
        return None

# ---------------- TTS ----------------
def synthesize_tts(text: str, index: int):
    voice = choose_voice_for_text(text)
    fname = f"audio_{index}.mp3"
    async def run_tts():
        comm = edge_tts.Communicate(text, voice)
        await comm.save(fname)
    try:
        try: 
            loop = asyncio.get_event_loop()
        except RuntimeError: 
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(run_tts())
        
        # Verify audio file exists
        if os.path.exists(fname) and os.path.getsize(fname) > 100:
            return fname
        else:
            st.warning(f"⚠️ Audio file for scene {index+1} seems invalid")
            return None
    except Exception as e:
        st.error(f"TTS Error for scene {index+1}: {e}")
        return None

# ---------------- Video Build ----------------
def create_video(script, topic):
    clips = []
    progress_bar = st.progress(0)
    status = st.empty()
    
    total_scenes = len(script)
    st.info(f"📊 Processing {total_scenes} scenes...")
    
    for i, scene in enumerate(script):
        status.markdown(f"**🎬 Generating Scene {i+1}/{total_scenes}...**")
        
        # 🔧 FIX: Simplified and more robust scene data extraction
        if not isinstance(scene, dict):
            st.error(f"Scene {i+1} is not a dictionary: {type(scene)}")
            continue
            
        narration = scene.get("narration", "").strip()
        visual_desc = scene.get("image_prompt", "").strip()
        
        if not narration or not visual_desc:
            st.error(f"Scene {i+1} missing narration or image_prompt")
            continue
        
        st.caption(f"📝 Narration: {narration[:80]}...")
        st.caption(f"🎨 Visual: {visual_desc[:80]}...")
        
        # 1. Determine Generation Strategy
        ref_path = (get_character_reference_from_topic(visual_desc) or 
                   get_character_reference_from_topic(narration) or 
                   get_character_reference_from_topic(topic))
        
        if ref_path:
            st.caption(f"🎭 Using Reference Context: {os.path.basename(ref_path)}")
        else:
            st.caption("🎨 Generating New Concept (No Reference)")
            
        # 2. Generate Image
        img_file = generate_scene_image_sd(visual_desc, ref_path, i)
            
        if not img_file:
            st.error(f"❌ Failed to generate image for scene {i+1}")
            continue
        
        # 3. Add Subtitles (Burn-in)
        img_with_sub = add_subtitles_to_image(img_file, narration)
        
        # 4. Generate Audio
        audio_file = synthesize_tts(narration, i)
        if not audio_file:
            st.error(f"❌ Failed to generate audio for scene {i+1}")
            continue
        
        # 5. Create Video Clip
        try:
            ac = AudioFileClip(audio_file)
            
            # Ensure minimum duration of 3 seconds
            duration = max(ac.duration, 3.0)
            
            # Try moviepy v2.0+ syntax first, fallback to v1.x
            try:
                vc = (ImageClip(img_with_sub)
                     .with_duration(duration)
                     .with_audio(ac)
                     .with_fps(24))
            except AttributeError:
                vc = (ImageClip(img_with_sub)
                     .set_duration(duration)
                     .set_audio(ac)
                     .set_fps(24))
            
            clips.append(vc)
            st.success(f"✅ Scene {i+1} complete!")
            
        except Exception as e:
            st.error(f"❌ Clip creation error for scene {i+1}: {e}")
            
        progress_bar.progress((i+1)/total_scenes)

    if not clips:
        st.error("❌ No clips were created successfully")
        return None
    
    if len(clips) < total_scenes:
        st.warning(f"⚠️ Only {len(clips)}/{total_scenes} scenes were created successfully")

    status.text("✨ Rendering Final Masterpiece...")
    output_filename = "mythos_masterpiece.mp4"
    
    try:
        final_video = concatenate_videoclips(clips, method="compose")
        final_video.write_videofile(
            output_filename, 
            codec="libx264", 
            audio_codec="aac", 
            fps=24,
            logger=None
        )
        
        # Cleanup
        final_video.close()
        for c in clips:
            c.close()
            if c.audio:
                c.audio.close()
        
        st.success(f"🎉 Video created with {len(clips)} scenes!")
        return output_filename
        
    except Exception as e:
        st.error(f"❌ Video rendering error: {e}")
        return None

# ---------------- UI ----------------
topic = st.text_input(
    "Enter your Mythological Story Idea (Hindi/English)", 
    placeholder="e.g. Ram and Sita in the forest, or अर्जुन युद्धभूमि में प्रवेश करता है"
)

if st.button("🚀 Generate Epic Video"):
    if not topic:
        st.warning("Please enter a topic!")
        st.stop()
    
    # Generate script with retry logic
    script = None
    max_script_attempts = 3
    
    with st.spinner("📜 Writing Ancient Scripture (Scripting)..."):
        for attempt in range(max_script_attempts):
            script = generate_script_gpt4(topic)
            if script and len(script) == 4:
                break
            if attempt < max_script_attempts - 1:
                st.warning(f"Retrying script generation (attempt {attempt+2}/{max_script_attempts})...")
                time.sleep(2)
    
    if not script or len(script) != 4:
        st.error("❌ Failed to generate valid 4-scene script after multiple attempts")
        st.stop()
    
    st.success(f"✅ Script generated with {len(script)} scenes")
    
    with st.expander("📖 View Script", expanded=False):
        st.json(script)
    
    # Generate video
    video_path = create_video(script, topic)
    
    if video_path and os.path.exists(video_path):
        st.success("🎉 Your Mythological Saga is Ready!")
        st.video(video_path)
        with open(video_path, "rb") as f:
            st.download_button(
                "⬇️ Download Video", 
                f, 
                file_name="mythos_saga.mp4",
                mime="video/mp4"
            )
    else:
        st.error("❌ Video generation failed. Please try again.")