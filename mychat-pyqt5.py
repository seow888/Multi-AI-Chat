import sys
import os
import json
import yaml
import logging
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QPushButton,
    QTextEdit, QTextBrowser, QFileDialog, QDialog, QGridLayout, QMessageBox, QLineEdit, QComboBox, QFrame, QListWidgetItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QKeySequence, QTextCursor
import google.generativeai as genai
from openai import OpenAI
import anthropic
import requests
import subprocess

# Configure logging
logging.basicConfig(filename='mychat.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the chat_logs directory exists
if not os.path.exists("chat_logs"):
    os.makedirs("chat_logs")

class APIConfigManager:
    """Manages API configurations."""
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        """Loads configuration from config.yaml."""
        try:
            with open("config.yaml", "r") as file:
                return yaml.safe_load(file) or {}
        except FileNotFoundError:
            return {}
        except Exception as e:
            logging.error(f"Error loading config.yaml file: {e}")
            return {}

    def save_config(self):
        """Saves configuration to config.yaml."""
        try:
            with open("config.yaml", "w") as file:
                yaml.dump(self.config, file)
        except Exception as e:
            logging.error(f"Error saving to config.yaml file: {e}")

    def get_active_provider(self):
        """Returns the active API provider."""
        return self.config.get("active_provider", "")

    def get_provider_config(self, provider):
        """Returns the configuration for a specific provider."""
        return self.config.get(provider, {})

class ChatSessionManager:
    """Manages chat sessions."""
    def __init__(self):
        self.sessions = {}  # {session_id: {"session_name": "", "chat_log": [], "conversation_history": [], "attached_files": {}}}
        self.load_sessions()

    def load_sessions(self):
        """Loads chat sessions from the chat_logs directory."""
        for filename in os.listdir("chat_logs"):
            if filename.endswith(".json"):
                session_id = filename[:-5]
                try:
                    with open(os.path.join("chat_logs", filename), "r") as file:
                        session_data = json.load(file)
                        # Validate session data
                        if not self.validate_session_data(session_data):
                            logging.error(f"Invalid session data in file: {filename}")
                            continue
                        self.sessions[session_id] = session_data
                except json.JSONDecodeError:
                    logging.error(f"Invalid JSON in file: {filename}")
                    continue
                except Exception as e:
                    logging.error(f"Error loading session file {filename}: {e}")
                    continue

    def validate_session_data(self, session_data):
        """Validates session data to ensure required fields exist."""
        required_fields = ["session_name", "chat_log", "conversation_history", "attached_files"]
        return all(field in session_data for field in required_fields)

    def convert_conversation_history(self, conversation_history, provider, system_prompt):
        """Converts conversation history to the format required by the selected provider."""
        converted_history = []

        if provider == "Google Gemini":
            for message in conversation_history:
                if message["role"] == "user":
                    converted_history.append({"role": "user", "parts": [{"text": message["content"]}]})
                elif message["role"] == "assistant" or message["role"] == "model":
                    converted_history.append({"role": "model", "parts": [{"text": message["content"]}]})
        elif provider in ["OpenAI", "Anthropic Claude", "xAI Grok", "Ollama"]:
            # Add system prompt for other providers
            converted_history.append({"role": "system", "content": system_prompt})
            for message in conversation_history:
                converted_history.append({"role": message["role"], "content": message["content"]})
        return converted_history

    def new_session(self):
        """Creates a new chat session."""
        timestamp = datetime.now()
        session_id = timestamp.strftime("Session_%Y%m%d_%H%M%S")
        session_name = timestamp.strftime("Session %Y-%m-%d %H:%M:%S")
        self.sessions[session_id] = {
            "session_name": session_name,
            "chat_log": [],
            "conversation_history": [],
            "attached_files": {}
        }
        return session_id, session_name

    def delete_session(self, session_id):
        """Deletes a chat session."""
        if session_id in self.sessions:
            try:
                log_file_path = os.path.join("chat_logs", f"{session_id}.json")
                if os.path.exists(log_file_path):
                    os.remove(log_file_path)
                del self.sessions[session_id]
            except Exception as e:
                logging.error(f"Error deleting session {session_id} and it's log file: {e}")

    def add_message(self, session_id, sender, message, timestamp, api_provider):
        """Adds a message to a chat session."""
        if session_id in self.sessions:
            self.sessions[session_id]["chat_log"].append((sender, message, timestamp))
            if sender == "You":
                self.sessions[session_id]["conversation_history"].append({"role": "user", "content": message})
            elif sender == "AI":
                self.sessions[session_id]["conversation_history"].append({"role": "assistant", "content": message})

    def get_session_messages(self, session_id):
        """Returns the messages for a specific session."""
        return self.sessions.get(session_id, {}).get("chat_log", [])

    def attach_file(self, session_id, file_path):
        """Attaches a file to a chat session."""
        if session_id in self.sessions:
            file_name = os.path.basename(file_path)
            try:
               with open(file_path, "rb") as file:
                   file_content = file.read()
               self.sessions[session_id]["attached_files"][file_name] = file_content
               return file_name
            except Exception as e:
              logging.error(f"Failed to attach file {file_path}: {e}")
              raise

    def get_attached_files(self, session_id):
        """Returns the attached files for a specific session."""
        return self.sessions.get(session_id, {}).get("attached_files", {})

    def save_session(self, session_id):
        """Saves a chat session to a file."""
        if session_id in self.sessions:
            file_path = os.path.join("chat_logs", f"{session_id}.json")
            try:
                with open(file_path, "w") as file:
                   json.dump({
                       "session_name": self.sessions[session_id]["session_name"],
                       "chat_log": self.sessions[session_id]["chat_log"],
                       "conversation_history": self.sessions[session_id]["conversation_history"],
                       "attached_files": self.sessions[session_id]["attached_files"]
                        }, file, indent=4)
            except Exception as e:
                 logging.error(f"Error saving session {session_id} to file {file_path}: {e}")

class MessageManager:
    """Manages message formatting and display."""
    def __init__(self, chat_display):
        self.chat_display = chat_display

    def add_formatted_message(self, sender, message, timestamp, bg_color):
        """Adds a formatted message to the chat display."""
        formatted_message = self.format_message(message)
        self.chat_display.append(f"<div style='background-color:{bg_color}; padding:5px; border-radius:5px;'>"
                                f"<b>{timestamp} {sender}:</b> {formatted_message}</div>")

    def format_message(self, message):
        """Formats the message with bold, italic, underline, and code block."""
        formatted_text = message.replace("*", "<b>").replace("*", "</b>")
        formatted_text = formatted_text.replace("_", "<i>").replace("_", "</i>")
        formatted_text = formatted_text.replace("~", "<u>").replace("~", "</u>")
        formatted_text = formatted_text.replace("``````", "</pre>")
        return formatted_text

class ApiWorker(QThread):
    """Handles asynchronous API requests."""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, session_id, user_message, parent, api_config_manager, chat_session_manager):
        super().__init__(parent)
        self.session_id = session_id
        self.user_message = user_message
        self.api_config_manager = api_config_manager
        self.chat_session_manager = chat_session_manager

    def run(self):
        """Generates the AI response."""
        max_retries = 3
        retry_delay = 2
        ai_message = ""
        for attempt in range(max_retries):
            try:
                provider = self.api_config_manager.get_active_provider()
                provider_config = self.api_config_manager.get_provider_config(provider)
                conversation_history = self.chat_session_manager.sessions[self.session_id]["conversation_history"]
                system_prompt = provider_config.get("system_prompt", "You are a helpful assistant.")

                # Convert conversation history to the format required by the selected provider
                converted_history = self.chat_session_manager.convert_conversation_history(conversation_history, provider, system_prompt)

                if provider == "Google Gemini":
                    genai.configure(api_key=provider_config.get("api_key", ""))
                    model = genai.GenerativeModel(provider_config.get("model", "gemini-pro"))
                    response = model.generate_content(converted_history)
                    ai_message = response.text

                elif provider == "OpenAI":
                    client = OpenAI(api_key=provider_config.get("api_key", ""))
                    response = client.chat.completions.create(
                        model=provider_config.get("model", "gpt-4"),
                        messages=converted_history,
                        temperature=0.7,
                        max_tokens=10000
                    )
                    ai_message = response.choices[0].message.content

                elif provider == "Anthropic Claude":
                    client = anthropic.Anthropic(api_key=provider_config.get("api_key", ""))
                    response = client.messages.create(
                        model=provider_config.get("model", "claude-3-sonnet-20240229"),
                        messages=converted_history,
                        temperature=0.7,
                        max_tokens=1000
                    )
                    ai_message = next(content.text for content in response.content if content.type == "text")

                elif provider == "xAI Grok":
                    url = "https://api.x.ai/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {provider_config.get('api_key', '')}",
                        "Content-Type": "application/json"
                    }
                    data = {
                        "model": "grok-2-1212",  # Updated model name
                        "messages": converted_history,
                        "temperature": 0.7,
                        "max_tokens": 10000,
                    }
                    response = requests.post(url, headers=headers, json=data)
                    if response.status_code != 200:
                        if response.status_code == 429 and attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        logging.error(f"xAI API Error: {response.text}")
                        raise Exception(f"API Error: {response.text}")
                    ai_message = response.json()['choices'][0]['message']['content']

                elif provider == "Ollama":
                    process = subprocess.Popen(
                        ["ollama", "run", provider_config.get("ollama_model", "llama2")],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    stdout, stderr = process.communicate(input=self.user_message)
                    if stderr:
                        raise Exception(stderr)
                    ai_message = stdout

                else:
                    raise Exception("Unsupported API provider")
                self.finished.emit(ai_message)
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logging.error(f"API call failed: {e}")
                    self.error.emit(str(e))

class APIConfigDialog(QDialog):
    """Dialog for configuring API settings."""
    def __init__(self, api_config_manager, parent=None):
        super().__init__(parent)
        self.api_config_manager = api_config_manager
        self.setWindowTitle("API Configuration")
        self.setGeometry(300, 300, 400, 300)
        layout = QVBoxLayout()

        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["Google Gemini", "OpenAI", "Anthropic Claude", "Ollama", "xAI Grok"])
        layout.addWidget(QLabel("API Provider:"))
        layout.addWidget(self.provider_combo)

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(QLabel("API Key:"))
        layout.addWidget(self.api_key_input)

        self.base_url_input = QLineEdit()
        layout.addWidget(QLabel("Base URL:"))
        layout.addWidget(self.base_url_input)

        self.model_input = QLineEdit()
        layout.addWidget(QLabel("Model:"))
        layout.addWidget(self.model_input)

        self.system_prompt_input = QLineEdit()
        layout.addWidget(QLabel("System Prompt:"))
        layout.addWidget(self.system_prompt_input)

        self.temperature_input = QLineEdit()
        layout.addWidget(QLabel("Temperature:"))
        layout.addWidget(self.temperature_input)

        self.ollama_model_combo = QComboBox()
        layout.addWidget(QLabel("Ollama Model:"))
        layout.addWidget(self.ollama_model_combo)
        self.update_ollama_models()

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_config)
        layout.addWidget(self.save_button)

        self.setLayout(layout)
        self.provider_combo.currentTextChanged.connect(self.update_fields)
        self.update_fields()

    def update_ollama_models(self):
        """Fetches Ollama models and updates the combo box."""
        try:
            process = subprocess.run(["ollama", "list"],
                                   capture_output=True,
                                   text=True,
                                   check=True)

            models = [line.split()[0] for line in process.stdout.strip().split("\n")
                     if line.strip()]

            self.ollama_model_combo.clear()
            self.ollama_model_combo.addItems(models)

            # Select currently configured model if it exists
            current_model = self.api_config_manager.config.get("Ollama", {}).get("ollama_model")
            if current_model in models:
                self.ollama_model_combo.setCurrentText(current_model)

        except subprocess.CalledProcessError as e:
            QMessageBox.warning(self, "Warning",
                              f"Failed to fetch Ollama models: {str(e)}")
        except Exception as e:
            QMessageBox.warning(self, "Warning",
                              f"Error updating Ollama models: {str(e)}")

    def update_fields(self):
        """Updates the fields based on the selected provider."""
        provider = self.provider_combo.currentText()
        provider_config = self.api_config_manager.get_provider_config(provider)
        self.api_key_input.setText(provider_config.get("api_key", ""))
        self.base_url_input.setText(provider_config.get("base_url", ""))
        self.model_input.setText(provider_config.get("model", ""))
        self.system_prompt_input.setText(provider_config.get("system_prompt", ""))
        self.temperature_input.setText(str(provider_config.get("temperature", "")))
        self.ollama_model_combo.setCurrentText(provider_config.get("ollama_model",""))
        self.update_ollama_fields_visibility(provider)

    def update_ollama_fields_visibility(self, provider):
        """Shows/hides Ollama-specific fields based on the provider."""
        ollama_visible = provider == "Ollama"
        self.ollama_model_combo.setVisible(ollama_visible)
        self.layout().itemAt(10).widget().setVisible(ollama_visible) # Label above ollama model dropdown

    def save_config(self):
        """Saves the API configuration."""
        provider = self.provider_combo.currentText()
        
        # Validate xAI Grok URL if applicable
        if provider == "xAI Grok" and not self.validate_grok_url(self.base_url_input.text()):
            QMessageBox.critical(self, "Error",
                               "Grok Base URL must point to https://api.x.ai")
            return
            
        config = {
            "api_key": self.api_key_input.text(),
            "base_url": self.base_url_input.text(),
            "model": self.model_input.text(),
            "system_prompt": self.system_prompt_input.text(),
            "temperature": float(self.temperature_input.text()) if self.temperature_input.text() else 0.7,
        }
        
        if provider == "Ollama":
            config["ollama_model"] = self.ollama_model_combo.currentText()
            
        self.api_config_manager.config[provider] = config
        self.api_config_manager.config["active_provider"] = provider
        self.api_config_manager.save_config()
        
        QMessageBox.information(self, "Success", "API configuration saved successfully!")
        self.accept()

    def validate_grok_url(self, url):
        """Validates the xAI Grok base URL."""
        return url.startswith("https://api.x.ai")

class MainWindow(QMainWindow):
    """Main application window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MyChat App")
        self.setGeometry(100, 100, 800, 600)

        # Color palette
        self.left_bg = "#203354"
        self.left_text = "#E5E7EB"
        self.button_bg = "#38B2AC"
        self.button_text = "#FFFFFF"
        self.right_bg = "#F6F9FC"
        self.user_msg_bg = "#B2D8D8"
        self.ai_msg_bg = "#E3E4FA"
        self.right_text = "#1A202C"
        self.highlight_color = "#FF6F61"
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {self.right_bg};}}
            QListWidget {{ background-color: {self.left_bg}; color: {self.left_text}; selection-background-color: {self.highlight_color}; selection-color: {self.left_text};}}
            QLabel {{ color: {self.right_text}; }}
            QTextBrowser {{ background-color: {self.right_bg}; color: {self.right_text}; }}
            QTextEdit {{ background-color: {self.right_bg}; color: {self.right_text}; }}
            QPushButton {{ background-color: {self.button_bg}; color: {self.button_text}; }}
        """)

        # Initialize API Config Manager and Chat Session Manager
        self.api_config_manager = APIConfigManager()
        self.chat_session_manager = ChatSessionManager()
        self.current_session_id = None

        # Initialize UI Elements
        self.create_menu_bar()
        self.create_main_layout()
        self.load_previous_sessions()
        if not self.chat_session_manager.sessions:
            self.new_session()
        else:
            try:
                sessions = sorted(
                    self.chat_session_manager.sessions.items(),
                    key=lambda x: os.path.getmtime(os.path.join("chat_logs", f"{x[0]}.json")),
                    reverse=True
                )
                if sessions:
                    self.load_session(sessions[0][0])
            except Exception as e:
                logging.error(f"Error loading the last session: {e}")
        self.initialize_api()
        self.update_api_label()

        # Initialize Message Manager
        self.initialize_message_manager()

    def initialize_message_manager(self):
        """Initializes the message manager."""
        self.message_manager = MessageManager(self.chat_display)

    def create_menu_bar(self):
        """Creates the menu bar."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        file_menu.addAction("New Session", self.new_session)
        file_menu.addAction("Export Chat", self.export_chat)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        settings_menu = menu_bar.addMenu("Settings")
        settings_menu.addAction("API Configuration", self.show_api_config)

    def create_main_layout(self):
        """Creates the main layout."""
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # Session Panel
        self.session_frame = QFrame()
        self.session_frame.setStyleSheet(f"background-color: {self.left_bg};")
        session_layout = QVBoxLayout(self.session_frame)
        self.session_label = QLabel("Sessions")
        self.session_label.setStyleSheet(f"color: {self.left_text}; font-weight: bold;")
        self.session_list_widget = QListWidget()
        self.session_list_widget.itemDoubleClicked.connect(self.load_selected_session)
        self.session_list_widget.currentItemChanged.connect(self.handle_session_selection)
        self.new_session_button = QPushButton("New Session")
        self.new_session_button.clicked.connect(self.new_session)
        self.delete_session_button = QPushButton("Delete Session")
        self.delete_session_button.clicked.connect(self.delete_session)
        session_layout.addWidget(self.session_label)
        session_layout.addWidget(self.session_list_widget)
        session_layout.addWidget(self.new_session_button)
        session_layout.addWidget(self.delete_session_button)
        main_layout.addWidget(self.session_frame, 1)

        # Chat Panel
        self.chat_frame = QFrame()
        self.chat_frame.setStyleSheet(f"background-color: {self.right_bg};")
        chat_layout = QVBoxLayout(self.chat_frame)
        self.api_label = QLabel("Current API: None")
        self.api_label.setStyleSheet(f"color: {self.right_text}; font-style: italic;")
        self.chat_display = QTextBrowser()
        self.chat_display.setOpenExternalLinks(True)
        self.message_input = QTextEdit()
        self.message_input.setFixedHeight(70)
        self.message_input.keyPressEvent = self.handle_key_press
        button_style = f"""
            QPushButton {{
                background-color: {self.button_bg};
                color: {self.button_text};
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.highlight_color};
            }}
        """
        self.send_button = QPushButton("Send")
        self.attach_button = QPushButton("Attach File")
        self.emoji_button = QPushButton("😊 Emoji")
        for button in [self.send_button, self.attach_button, self.emoji_button]:
            button.setStyleSheet(button_style)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.send_button)
        button_layout.addWidget(self.attach_button)
        button_layout.addWidget(self.emoji_button)
        button_layout.setSpacing(10)
        chat_layout.addWidget(self.api_label)
        chat_layout.addWidget(self.chat_display)
        chat_layout.addWidget(self.message_input)
        chat_layout.addLayout(button_layout)
        main_layout.addWidget(self.chat_frame, 3)
        self.send_button.clicked.connect(self.send_message)
        self.attach_button.clicked.connect(self.attach_file)
        self.emoji_button.clicked.connect(self.insert_emoji)

    def insert_emoji(self):
        """Opens the emoji selector window."""
        emoji_dialog = QDialog(self)
        emoji_dialog.setWindowTitle("Select Emoji")
        emoji_dialog.setGeometry(300, 300, 300, 200)
        layout = QGridLayout()
        emojis = ["😊", "😂", "👍", "❤️", "🎉", "😎", "🤔", "👋", "✨", "🔥"]

        def add_emoji(emoji):
            """Inserts the selected emoji into the message input."""
            self.message_input.insertPlainText(emoji)
            emoji_dialog.accept()

        for i, emoji in enumerate(emojis):
            button = QPushButton(emoji)
            button.setFont(QFont("Arial", 20))
            button.clicked.connect(lambda checked, e=emoji: add_emoji(e))
            layout.addWidget(button, i // 5, i % 5)

        emoji_dialog.setLayout(layout)
        emoji_dialog.exec_()

    def handle_key_press(self, event):
        """Handles key press events in the message input."""
        if event.key() == Qt.Key_Return and event.modifiers() == Qt.ShiftModifier:
            self.message_input.insertPlainText("\n")
        elif event.key() == Qt.Key_Return:
            self.send_message()
        elif event.modifiers() == Qt.ControlModifier:
            if event.key() == Qt.Key_B:
                self.toggle_bold()
            elif event.key() == Qt.Key_I:
                self.toggle_italic()
            elif event.key() == Qt.Key_U:
                self.toggle_underline()
        elif event.modifiers() == (Qt.ControlModifier | Qt.ShiftModifier) and event.key() == Qt.Key_K:
           self.toggle_code_block()
        else:
            QTextEdit.keyPressEvent(self.message_input, event)

    def toggle_bold(self):
        """Toggles bold formatting at the cursor position."""
        cursor = self.message_input.textCursor()
        if cursor.hasSelection():
            text = cursor.selectedText()
            cursor.insertText(f"<b>{text}</b>")

    def toggle_italic(self):
        """Toggles italic formatting at the cursor position."""
        cursor = self.message_input.textCursor()
        if cursor.hasSelection():
            text = cursor.selectedText()
            cursor.insertText(f"<i>{text}</i>")

    def toggle_underline(self):
        """Toggles underline formatting at the cursor position."""
        cursor = self.message_input.textCursor()
        if cursor.hasSelection():
            text = cursor.selectedText()
            cursor.insertText(f"<u>{text}</u>")

    def toggle_code_block(self):
        """Toggles code block formatting around the selected text."""
        cursor = self.message_input.textCursor()
        if cursor.hasSelection():
            selected_text = cursor.selectedText()
            cursor.insertText(f"<pre>{selected_text}</pre>")

    def load_config(self):
       """Loads the configuration from config.yaml."""
       self.config = self.api_config_manager.load_config()

    def save_config(self):
       """Saves the current configuration to config.yaml."""
       self.api_config_manager.save_config()

    def show_api_config(self):
        """Opens the API configuration window."""
        dialog = APIConfigDialog(self.api_config_manager, self)
        dialog.exec_()
        self.initialize_api()
        self.update_api_label()

    def initialize_api(self):
        """Initializes the API based on the active provider."""
        provider = self.api_config_manager.get_active_provider()
        if not provider:
            return
        provider_config = self.api_config_manager.get_provider_config(provider)
        try:
            if provider == "Google Gemini":
                genai.configure(api_key=provider_config.get("api_key", ""))
                self.model_instance = genai.GenerativeModel(provider_config.get("model", "gemini-pro"))
            elif provider in ["OpenAI", "OpenAI Compatible"]:
                self.openai_client = OpenAI(api_key=provider_config.get("api_key", ""))
                if provider == "OpenAI Compatible":
                    self.openai_client.base_url = provider_config.get("base_url", "")
            elif provider == "Anthropic Claude":
                self.anthropic_client = anthropic.Anthropic(api_key=provider_config.get("api_key", ""))
            elif provider == "xAI Grok":
                url = provider_config.get("base_url", "")
                if url and "https://api.x.ai" not in url:
                    QMessageBox.critical(self, "Error", "Grok Base URL must point to https://api.x.ai.")
                    return
            elif provider == "Ollama":
                self.ollama_model_name = provider_config.get("ollama_model", "llama2")
                self.ollama_command = ["ollama", "run", self.ollama_model_name]
        except Exception as e:
            QMessageBox.critical(self, "API Initialization Error", str(e))

    def update_api_label(self):
        """Updates the API label in the chat window."""
        provider = self.api_config_manager.get_active_provider()
        self.api_label.setText(f"Current API: {provider if provider else 'None'}")

    def export_chat(self):
        """Exports the chat log to a text file."""
        if not self.current_session_id:
            QMessageBox.warning(self, "Warning", "No active session to export.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Chat Log", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as file:
                    messages = self.chat_session_manager.get_session_messages(self.current_session_id)
                    for sender, message, timestamp in messages:
                        file.write(f"{timestamp} {sender}: {message}\n")
                    attachments = self.chat_session_manager.get_attached_files(self.current_session_id)
                    if attachments:
                        file.write("\n\nAttached Files:\n")
                        for file_name in attachments:
                            file.write(f"  {file_name}\n")
                QMessageBox.information(self, "Success", "Chat exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export chat log: {str(e)}")

    def new_session(self):
        """Creates a new chat session."""
        session_id, session_name = self.chat_session_manager.new_session()
        self.current_session_id = session_id
        self.chat_session_manager.save_session(self.current_session_id) # Save immediately
        self.update_session_list()
        self.update_chat_display()
        QMessageBox.information(self, "New Session", f"Session '{session_name}' created.")

    def delete_session(self):
        """Deletes the selected chat session."""
        selected_item = self.session_list_widget.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "Error", "Please select a session to delete.")
            return
        session_id = selected_item.data(Qt.UserRole)
        self.chat_session_manager.delete_session(session_id)
        self.current_session_id = None
        self.update_session_list()
        self.update_chat_display()
        QMessageBox.information(self, "Session Deleted", "Session deleted successfully.")

    def send_message(self):
       """Sends the user's message to the AI and updates the chat display."""
       message = self.message_input.toPlainText().strip()
       if not message:
           return
       if not self.current_session_id:
           QMessageBox.critical(self, "Error", "Please create or select a session first.")
           return

       timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
       provider = self.api_config_manager.get_active_provider()
       self.chat_session_manager.add_message(self.current_session_id, "You", message, timestamp, provider)
       self.chat_session_manager.save_session(self.current_session_id)
       self.message_input.clear()
       self.update_chat_display()
       self.start_api_worker(message)

    def start_api_worker(self, user_message):
        """Starts a new API worker thread to handle the AI request."""
        self.api_worker = ApiWorker(self.current_session_id, user_message, self, self.api_config_manager, self.chat_session_manager)
        self.api_worker.finished.connect(self.handle_ai_response)
        self.api_worker.error.connect(self.handle_api_error)
        self.api_worker.start()

    def handle_ai_response(self, ai_message):
        """Handles the AI's response and updates the chat display."""
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        provider = self.api_config_manager.get_active_provider()
        self.chat_session_manager.add_message(self.current_session_id, "AI", ai_message, timestamp, provider)
        self.update_chat_display()

    def handle_api_error(self, error_message):
        """Handles API errors and displays error messages."""
        QMessageBox.critical(self, "API Error", error_message)

    def update_session_list(self):
        """Updates the list of sessions in the left panel."""
        current_selection = self.session_list_widget.currentItem()
        current_session_id = current_selection.data(Qt.UserRole) if current_selection else None
        
        self.session_list_widget.clear()
        
        try:
            sessions = sorted(
                self.chat_session_manager.sessions.items(),
                key=lambda x: os.path.getmtime(os.path.join("chat_logs", f"{x[0]}.json")),
                reverse=True
            )
            
            for session_id, session_data in sessions:
                item = QListWidgetItem(session_data["session_name"])
                item.setData(Qt.UserRole, session_id)
                self.session_list_widget.addItem(item)
                
                # Restore selection if applicable
                if session_id == current_session_id:
                    self.session_list_widget.setCurrentItem(item)
                
        except OSError as e:
            logging.error(f"Error updating session list: {e}")
            QMessageBox.warning(self, "Warning", "Error updating session list")

    def handle_session_selection(self, current, previous):
        """Handles session selection changes."""
        if current:
            session_id = current.data(Qt.UserRole)
            self.load_session(session_id)

    def load_session(self, session_id):
        """Loads a session by ID."""
        if session_id in self.chat_session_manager.sessions:
            self.current_session_id = session_id
            self.initialize_message_manager()  # Reinitialize message manager
            self.update_chat_display()

    def load_selected_session(self, item):
        """Handles double-click on session items."""
        if item:
            session_id = item.data(Qt.UserRole)
            self.load_session(session_id)

    def update_chat_display(self):
        """Updates the chat display with the latest messages."""
        self.chat_display.clear()

        if not self.current_session_id:
            return

        try:
            messages = self.chat_session_manager.get_session_messages(self.current_session_id)

            for sender, message, timestamp in messages:
                bg_color = self.user_msg_bg if sender == "You" else self.ai_msg_bg
                self.message_manager.add_formatted_message(sender, message, timestamp, bg_color)

            # Display attachments if any
            attachments = self.chat_session_manager.get_attached_files(self.current_session_id)
            if attachments:
                self.chat_display.append("<br><br><b>Attached Files:</b><br>")
                for file_name in attachments:
                    self.chat_display.append(
                        f"  <a href='file://{os.path.abspath(os.path.join('chat_logs', f'{self.current_session_id}.json'))}'>"
                        f"{file_name}</a><br>"
                    )
        except Exception as e:
            logging.error(f"Error updating chat display: {e}")
            QMessageBox.warning(self, "Warning", "Error updating chat display")

    def load_previous_sessions(self):
       """Loads the previously saved sessions."""
       self.chat_session_manager.load_sessions()
       self.update_session_list()

    def attach_file(self):
        """Handles file attachment."""
        if not self.current_session_id:
            QMessageBox.critical(self, "Error", "Please create or select a session first.")
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "Attach File", "", "All Files (*)")
        if file_path:
            try:
                file_name = self.chat_session_manager.attach_file(self.current_session_id, file_path)
                timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                self.chat_session_manager.add_message(self.current_session_id, "System", f"File attached: {file_name}", timestamp, self.api_config_manager.get_active_provider())
                self.update_chat_display()
                self.chat_session_manager.save_session(self.current_session_id)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to attach file: {str(e)}")

if __name__ == "__main__":
    # Ensure the Qt platform plugin is set correctly
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

