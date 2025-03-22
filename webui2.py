# import os
# import torch
# import soundfile as sf
# import logging
# import argparse
# import gradio as gr
# import platform
# import json
# from datetime import datetime
# from cli.Model import SparkTTS
# from sparktts.utils.token_parser import LEVELS_MAP_UI

# # File to store user credentials
# USERS_FILE = "users.json"

# def load_users():
#     """Load users from the JSON file."""
#     if os.path.exists(USERS_FILE):
#         with open(USERS_FILE, 'r') as f:
#             return json.load(f)
#     return {"admin": "password123"}  # Default user if file doesn't exist

# def save_users(users):
#     """Save users to the JSON file."""
#     with open(USERS_FILE, 'w') as f:
#         json.dump(users, f, indent=4)

# def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device=0):
#     logging.info(f"Loading model from: {model_dir}")
#     device = torch.device("cpu")
#     logging.info("Using CPU device explicitly")
#     model = SparkTTS(model_dir, device)
#     return model

# def run_tts(
#     text,
#     model,
#     prompt_text=None,
#     prompt_speech=None,
#     gender=None,
#     pitch=None,
#     speed=None,
#     save_dir="example/results",
# ):
#     """Perform TTS inference and save the generated audio."""
#     logging.info(f"Saving audio to: {save_dir}")
#     if prompt_text is not None:
#         prompt_text = None if len(prompt_text) <= 1 else prompt_text
#     os.makedirs(save_dir, exist_ok=True)
#     timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#     save_path = os.path.join(save_dir, f"{timestamp}.wav")
#     logging.info("Starting inference...")
#     with torch.no_grad():
#         wav = model.inference(
#             text,
#             prompt_speech,
#             prompt_text,
#             gender,
#             pitch,
#             speed,
#         )
#         sf.write(save_path, wav, samplerate=16000)
#     logging.info(f"Audio saved at: {save_path}")
#     return save_path

# def authenticate(username, password, is_signup=False):
#     """Handle login and signup logic with JSON storage."""
#     users = load_users()
#     if is_signup:
#         if username in users:
#             return "Username already exists. Please choose a different one.", False
#         if not username or not password:
#             return "Username and password cannot be empty.", False
#         users[username] = password
#         save_users(users)
#         logging.info(f"User {username} signed up successfully.")
#         return "Signup successful! Please log in.", True
#     else:
#         if username in users and users[username] == password:
#             logging.info(f"User {username} logged in successfully.")
#             return "Login successful!", True
#         return "Invalid username or password.", False

# def build_ui(model_dir, device=0):
#     model = initialize_model(model_dir, device=device)

#     def voice_clone(text, prompt_text, prompt_wav_upload, prompt_wav_record):
#         prompt_speech = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
#         prompt_text_clean = None if len(prompt_text) < 2 else prompt_text
#         audio_output_path = run_tts(
#             text,
#             model,
#             prompt_text=prompt_text_clean,
#             prompt_speech=prompt_speech
#         )
#         return audio_output_path

#     def voice_creation(text, gender, pitch, speed):
#         pitch_val = LEVELS_MAP_UI[int(pitch)]
#         speed_val = LEVELS_MAP_UI[int(speed)]
#         audio_output_path = run_tts(
#             text,
#             model,
#             gender=gender,
#             pitch=pitch_val,
#             speed=speed_val
#         )
#         return audio_output_path

#     def main_interface():
#         with gr.Blocks() as main_demo:
#             gr.HTML('<h1 style="text-align: center;">Syndicate Smashers</h1>')
#             with gr.Tabs():
#                 with gr.TabItem("Voice Clone"):
#                     gr.Markdown("### Upload reference audio or recording")
#                     with gr.Row():
#                         prompt_wav_upload = gr.Audio(
#                             sources="upload",
#                             type="filepath",
#                             label="Choose the prompt audio file, ensuring the sampling rate is no lower than 16kHz.",
#                         )
#                         prompt_wav_record = gr.Audio(
#                             sources="microphone",
#                             type="filepath",
#                             label="Record the prompt audio file.",
#                         )
#                     with gr.Row():
#                         text_input = gr.Textbox(
#                             label="Text", lines=3, placeholder="Enter text here"
#                         )
#                         prompt_text_input = gr.Textbox(
#                             label="Text of prompt speech (Optional; recommended for cloning in the same language.)",
#                             lines=3,
#                             placeholder="Enter text of the prompt speech.",
#                         )
#                     audio_output = gr.Audio(
#                         label="Generated Audio", autoplay=True, streaming=True
#                     )
#                     generate_button_clone = gr.Button("Generate")
#                     generate_button_clone.click(
#                         voice_clone,
#                         inputs=[text_input, prompt_text_input, prompt_wav_upload, prompt_wav_record],
#                         outputs=[audio_output],
#                     )

#                 with gr.TabItem("Voice Creation"):
#                     gr.Markdown("### Create your own voice based on the following parameters")
#                     with gr.Row():
#                         with gr.Column():
#                             gender = gr.Radio(
#                                 choices=["male", "female"], value="male", label="Gender"
#                             )
#                             pitch = gr.Slider(
#                                 minimum=1, maximum=5, step=1, value=3, label="Pitch"
#                             )
#                             speed = gr.Slider(
#                                 minimum=1, maximum=5, step=1, value=3, label="Speed"
#                             )
#                         with gr.Column():
#                             text_input_creation = gr.Textbox(
#                                 label="Input Text",
#                                 lines=3,
#                                 placeholder="Enter text here",
#                                 value="Welcome to Syndicate Smashers TTS model.",
#                             )
#                             create_button = gr.Button("Create Voice")
#                     audio_output = gr.Audio(
#                         label="Generated Audio", autoplay=True, streaming=True
#                     )
#                     create_button.click(
#                         voice_creation,
#                         inputs=[text_input_creation, gender, pitch, speed],
#                         outputs=[audio_output],
#                     )
#         return main_demo

#     with gr.Blocks() as demo:
#         gr.HTML('<h1 style="text-align: center;">Welcome to Syndicate Smashers</h1>')
#         with gr.Tabs():
#             with gr.TabItem("Login"):
#                 username_input = gr.Textbox(label="Username", placeholder="Enter username")
#                 password_input = gr.Textbox(label="Password", type="password", placeholder="Enter password")
#                 login_button = gr.Button("Login")
#                 login_output = gr.Textbox(label="Status")
#                 login_success = gr.State(value=False)

#                 def login_handler(username, password):
#                     message, success = authenticate(username, password)
#                     return message, success

#                 login_button.click(
#                     login_handler,
#                     inputs=[username_input, password_input],
#                     outputs=[login_output, login_success]
#                 )

#             with gr.TabItem("Sign Up"):
#                 signup_username = gr.Textbox(label="Username", placeholder="Choose a username")
#                 signup_password = gr.Textbox(label="Password", type="password", placeholder="Choose a password")
#                 signup_button = gr.Button("Sign Up")
#                 signup_output = gr.Textbox(label="Status")

#                 def signup_handler(username, password):
#                     message, _ = authenticate(username, password, is_signup=True)
#                     return message

#                 signup_button.click(
#                     signup_handler,
#                     inputs=[signup_username, signup_password],
#                     outputs=[signup_output]
#                 )

#         # Main interface will be shown only after successful login
#         with gr.Row(visible=False) as main_row:
#             main_demo = main_interface()

#         def show_main_interface(login_success):
#             return gr.Row(visible=login_success)

#         login_success.change(
#             show_main_interface,
#             inputs=[login_success],
#             outputs=[main_row]
#         )

#     return demo

# def parse_arguments():
#     parser = argparse.ArgumentParser(description="Spark TTS Gradio server.")
#     parser.add_argument(
#         "--model_dir",
#         type=str,
#         default="pretrained_models/Spark-TTS-0.5B",
#         help="Path to the model directory."
#     )
#     parser.add_argument(
#         "--device",
#         type=int,
#         default=0,
#         help="ID of the GPU device to use (e.g., 0 for cuda:0)."
#     )
#     parser.add_argument(
#         "--server_name",
#         type=str,
#         default="0.0.0.0",
#         help="Server host/IP for Gradio app."
#     )
#     parser.add_argument(
#         "--server_port",
#         type=int,
#         default=7860,
#         help="Server port for Gradio app."
#     )
#     return parser.parse_args()

# if __name__ == "__main__":
#     args = parse_arguments()
#     demo = build_ui(
#         model_dir=args.model_dir,
#         device=args.device
#     )
#     demo.launch(
#         server_name=args.server_name,
#         server_port=args.server_port
#     )



import os
import torch
import soundfile as sf
import logging
import argparse
import gradio as gr
import platform
import json
import bcrypt
from datetime import datetime
from cli.Model import SparkTTS
from sparktts.utils.token_parser import LEVELS_MAP_UI

# File to store user credentials
USER_FILE = "user.json"

def load_users():
    """Load users from user.json, or return an empty dict if the file doesn't exist."""
    if os.path.exists(USER_FILE):
        with open(USER_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to user.json."""
    with open(USER_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device=0):
    logging.info(f"Loading model from: {model_dir}")
    device = torch.device("cpu")
    logging.info("Using CPU device explicitly")
    model = SparkTTS(model_dir, device)
    return model

def run_tts(
    text,
    model,
    prompt_text=None,
    prompt_speech=None,
    gender=None,
    pitch=None,
    speed=None,
    save_dir="example/results",
):
    """Perform TTS inference and save the generated audio."""
    logging.info(f"Saving audio to: {save_dir}")
    if prompt_text is not None:
        prompt_text = None if len(prompt_text) <= 1 else prompt_text
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}.wav")
    logging.info("Starting inference...")
    with torch.no_grad():
        wav = model.inference(
            text,
            prompt_speech,
            prompt_text,
            gender,
            pitch,
            speed,
        )
        sf.write(save_path, wav, samplerate=16000)
    logging.info(f"Audio saved at: {save_path}")
    return save_path

def authenticate(username, password, is_signup=False):
    """Handle login and signup logic with encrypted passwords."""
    users = load_users()
    
    if is_signup:
        if username in users:
            return "Username already exists. Please choose a different one.", False
        if not username or not password:
            return "Username and password cannot be empty.", False
        # Hash the password with bcrypt
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        users[username] = hashed_password.decode('utf-8')  # Store as string
        save_users(users)
        return "Signup successful! Please log in.", True
    else:
        if username in users:
            stored_hash = users[username].encode('utf-8')  # Convert stored hash back to bytes
            if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
                return "Login successful!", True
        return "Invalid username or password.", False

def build_login_ui():
    """Build the login/signup interface."""
    with gr.Blocks() as login_demo:
        gr.HTML('<h1 style="text-align: center;">Welcome to Syndicate Smashers</h1>')
        with gr.Tabs():
            with gr.TabItem("Login"):
                username_input = gr.Textbox(label="Username", placeholder="Enter username")
                password_input = gr.Textbox(label="Password", type="password", placeholder="Enter password")
                login_button = gr.Button("Login")
                login_output = gr.Textbox(label="Status")
                login_success = gr.State(value=False)

                def login_handler(username, password):
                    message, success = authenticate(username, password)
                    return message, success

                login_button.click(
                    login_handler,
                    inputs=[username_input, password_input],
                    outputs=[login_output, login_success]
                )

            with gr.TabItem("Sign Up"):
                signup_username = gr.Textbox(label="Username", placeholder="Choose a username")
                signup_password = gr.Textbox(label="Password", type="password", placeholder="Choose a password")
                signup_button = gr.Button("Sign Up")
                signup_output = gr.Textbox(label="Status")

                def signup_handler(username, password):
                    message, _ = authenticate(username, password, is_signup=True)
                    return message

                signup_button.click(
                    signup_handler,
                    inputs=[signup_username, signup_password],
                    outputs=[signup_output]
                )

    return login_demo, login_success

def build_main_ui(model_dir, device=0):
    """Build the main TTS interface."""
    model = initialize_model(model_dir, device=device)

    def voice_clone(text, prompt_text, prompt_wav_upload, prompt_wav_record):
        prompt_speech = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
        prompt_text_clean = None if len(prompt_text) < 2 else prompt_text
        audio_output_path = run_tts(
            text,
            model,
            prompt_text=prompt_text_clean,
            prompt_speech=prompt_speech
        )
        return audio_output_path

    def voice_creation(text, gender, pitch, speed):
        pitch_val = LEVELS_MAP_UI[int(pitch)]
        speed_val = LEVELS_MAP_UI[int(speed)]
        audio_output_path = run_tts(
            text,
            model,
            gender=gender,
            pitch=pitch_val,
            speed=speed_val
        )
        return audio_output_path

    with gr.Blocks() as main_demo:
        gr.HTML('<h1 style="text-align: center;">Syndicate Smashers TTS</h1>')
        with gr.Tabs():
            with gr.TabItem("Voice Clone"):
                gr.Markdown("### Upload reference audio or recording")
                with gr.Row():
                    prompt_wav_upload = gr.Audio(
                        sources="upload",
                        type="filepath",
                        label="Choose the prompt audio file, ensuring the sampling rate is no lower than 16kHz.",
                    )
                    prompt_wav_record = gr.Audio(
                        sources="microphone",
                        type="filepath",
                        label="Record the prompt audio file.",
                    )
                with gr.Row():
                    text_input = gr.Textbox(
                        label="Text", lines=3, placeholder="Enter text here"
                    )
                    prompt_text_input = gr.Textbox(
                        label="Text of prompt speech (Optional; recommended for cloning in the same language.)",
                        lines=3,
                        placeholder="Enter text of the prompt speech.",
                    )
                audio_output = gr.Audio(
                    label="Generated Audio", autoplay=True, streaming=True
                )
                generate_button_clone = gr.Button("Generate")
                generate_button_clone.click(
                    voice_clone,
                    inputs=[text_input, prompt_text_input, prompt_wav_upload, prompt_wav_record],
                    outputs=[audio_output],
                )

            with gr.TabItem("Voice Creation"):
                gr.Markdown("### Create your own voice based on the following parameters")
                with gr.Row():
                    with gr.Column():
                        gender = gr.Radio(
                            choices=["male", "female"], value="male", label="Gender"
                        )
                        pitch = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3, label="Pitch"
                        )
                        speed = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3, label="Speed"
                        )
                    with gr.Column():
                        text_input_creation = gr.Textbox(
                            label="Input Text",
                            lines=3,
                            placeholder="Enter text here",
                            value="Welcome to Syndicate Smashers TTS model.",
                        )
                        create_button = gr.Button("Create Voice")
                audio_output = gr.Audio(
                    label="Generated Audio", autoplay=True, streaming=True
                )
                create_button.click(
                    voice_creation,
                    inputs=[text_input_creation, gender, pitch, speed],
                    outputs=[audio_output],
                )

        logout_button = gr.Button("Logout")
        logout_success = gr.State(value=False)

        def logout_handler():
            return True

        logout_button.click(
            logout_handler,
            inputs=[],
            outputs=[logout_success]
        )

    return main_demo, logout_success

def build_app(model_dir, device=0):
    """Build the full application with login and main pages."""
    login_demo, login_success = build_login_ui()
    main_demo, logout_success = build_main_ui(model_dir, device)

    with gr.Blocks() as app:
        with gr.Column(visible=True) as login_page:
            login_demo.render()

        with gr.Column(visible=False) as main_page:
            main_demo.render()

        def switch_to_main(login_success):
            return gr.Column(visible=not login_success), gr.Column(visible=login_success)

        def switch_to_login(logout_success):
            return gr.Column(visible=logout_success), gr.Column(visible=not logout_success)

        login_success.change(
            switch_to_main,
            inputs=[login_success],
            outputs=[login_page, main_page]
        )

        logout_success.change(
            switch_to_login,
            inputs=[logout_success],
            outputs=[login_page, main_page]
        )

    return app

def parse_arguments():
    parser = argparse.ArgumentParser(description="Spark TTS Gradio server.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="Path to the model directory."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="ID of the GPU device to use (e.g., 0 for cuda:0)."
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="Server host/IP for Gradio app."
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="Server port for Gradio app."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    app = build_app(
        model_dir=args.model_dir,
        device=args.device
    )
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port
    )