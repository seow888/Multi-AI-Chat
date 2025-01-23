import sys
import os
import json
import yaml
import logging
import time
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QPushButton, QTextEdit, QTextBrowser, QFileDialog, QDialog,
    QGridLayout, QMessageBox, QLineEdit, QComboBox, QSlider, QMenuBar,
    QScrollArea, QDialogButtonBox, QSizePolicy, QMenu, QListWidgetItem, QDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDir, QSize
from PyQt6.QtGui import (
    QFont, QKeySequence, QTextCursor, QColor, QAction, QIcon,
    QTextCharFormat, QPalette, QGuiApplication
)
import google.generativeai as genai
from openai import OpenAI
import anthropic
import requests
import subprocess
import html
import markdown
import hashlib  # Added for file hashing

# Configure logging
logging.basicConfig(filename='mychat.log', level=logging.ERROR,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure directories exist
for dir in ["chat_logs", "config"]:
    if not os.path.exists(dir):
        os.makedirs(dir)

class APIConfigManager:
    """Manages API configurations with validation"""
    def __init__(self):
        self.config_path = os.path.join("config", "config.yaml")
        self.config = self.load_config()

    def load_config(self):
        """Load or create config with defaults"""
        try:
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file) or {}
                providers = ["Google Gemini", "OpenAI", "Anthropic Claude", "Ollama", "xAI Grok", "OpenAI-Compatible", "DeepSeek"]
                for provider in providers:
                    if provider not in config:
                        config[provider] = {}
                return config
        except Exception as e:
            logging.error(f"Config load error: {e}")
            return {
                "active_provider": "OpenAI",
                "Google Gemini": {"api_key": "", "model": "gemini-2.0-flash-exp", "temperature": 0.7},
                "OpenAI": {"api_key": "", "model": "gpt-4o-mini", "temperature": 0.7},
                "Anthropic Claude": {"api_key": "", "model": "claude-3-opus-20240229", "temperature": 0.7},
                "Ollama": {"ollama_model": "llama2", "temperature": 0.7},
                "xAI Grok": {"api_key": "", "base_url": "https://api.x.ai/v1", "temperature": 0.7},
                "OpenAI-Compatible": {"api_key": "", "base_url": "https://api.hyperbolic.xyz/v1", "model": "custom-model", "temperature": 0.7},
                "DeepSeek": {"api_key": "", "model": "deepseek-chat", "temperature": 0.7}
            }

    def save_config(self):
        """Save config with error handling"""
        try:
            with open(self.config_path, "w") as file:
                yaml.dump(self.config, file)
        except Exception as e:
            logging.error(f"Config save error: {e}")
            QMessageBox.critical(None, "Config Error", f"Failed to save config: {str(e)}")

    def get_active_provider(self):
        return self.config.get("active_provider", "OpenAI")

    def get_provider_config(self, provider):
        return self.config.get(provider, {})

    def set_active_provider(self, provider):
        self.config["active_provider"] = provider
        self.save_config()

class ChatSessionManager:
    """Manages chat sessions with file persistence"""
    def __init__(self):
        self.sessions = {}
        self.load_sessions()

    def load_sessions(self):
        """Load sessions from chat_logs directory"""
        try:
            for filename in os.listdir("chat_logs"):
                if filename.endswith(".json"):
                    session_id = filename[:-5]
                    file_path = os.path.join("chat_logs", filename)
                    try:
                        with open(file_path, "r") as file:
                            session_data = json.load(file)
                            if self.validate_session_data(session_data):
                                self.sessions[session_id] = session_data
                    except Exception as e:
                        logging.error(f"Error loading {filename}: {str(e)}")
        except Exception as e:
            logging.error(f"Session load error: {str(e)}")

    def validate_session_data(self, data):
        required = ["session_name", "chat_log", "conversation_history", "attached_files", "created_at"]
        return all(key in data for key in required)

    def convert_conversation_history(self, history, provider, system_prompt):
        """Convert history for different providers"""
        converted = []
        if provider == "Google Gemini":
            for msg in history:
                role = "model" if msg["role"] in ["assistant", "model"] else "user"
                converted.append({"role": role, "parts": [{"text": msg["content"]}]})
        else:
            converted.append({"role": "system", "content": system_prompt})
            for msg in history:
                converted.append({"role": msg["role"], "content": msg["content"]})
        return converted

    def new_session(self):
        """Create new session with timestamp"""
        session_id = datetime.now().strftime("Chat_%Y%m%d_%H%M%S")
        session_name = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.sessions[session_id] = {
            "session_name": session_name,
            "chat_log": [],
            "conversation_history": [],
            "attached_files": {},
            "created_at": datetime.now().timestamp()
        }
        self.save_session(session_id)
        return session_id, session_name

    def delete_session(self, session_id):
        """Improved with file check and error propagation"""
        if session_id in self.sessions:
            file_path = os.path.join("chat_logs", f"{session_id}.json")
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                del self.sessions[session_id]
            except Exception as e:
                logging.error(f"Error deleting session {session_id}: {str(e)}")
                raise  # Critical: Let UI handle this error

    def add_message(self, session_id, sender, message, timestamp):
        """Add message to session"""
        if session_id in self.sessions:
            self.sessions[session_id]["chat_log"].append((sender, message, timestamp))
            role = "user" if sender == "You" else "assistant"
            self.sessions[session_id]["conversation_history"].append({
                "role": role,
                "content": message
            })

    def save_session(self, session_id):
        """Save session to file"""
        if session_id in self.sessions:
            try:
                with open(os.path.join("chat_logs", f"{session_id}.json"), "w") as file:
                    json.dump(self.sessions[session_id], file, indent=4)
            except Exception as e:
                logging.error(f"Error saving session {session_id}: {str(e)}")

    def attach_file(self, session_id, file_path):
        """Store file metadata instead of content"""
        if session_id in self.sessions:
            try:
                if os.path.getsize(file_path) > 10 * 1024 * 1024:  # 10MB limit
                    raise ValueError("File size exceeds 10MB limit")
                
                file_name = os.path.basename(file_path)
                file_hash = self._calculate_file_hash(file_path)
                
                self.sessions[session_id]["attached_files"][file_name] = {
                    "path": file_path,
                    "hash": file_hash,
                    "timestamp": datetime.now().isoformat()
                }
                return file_name
            except Exception as e:
                logging.error(f"File attach error: {str(e)}")
                raise

    def _calculate_file_hash(self, file_path):
        """SHA256 hash for file verification"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(4096):
                hasher.update(chunk)
        return hasher.hexdigest()

class ModernMessageManager:
    """Handles message formatting with WhatsApp-like styling"""
    def __init__(self, chat_display):
        self.chat_display = chat_display
        self._setup_context_menu()
        self.styles = {
            "user": {"bg": "#DCF8C6", "text": "#000000", "border": "#B2D8A3"},
            "ai": {"bg": "#FFFFFF", "text": "#000000", "border": "#E0E0E0"},
            "system": {"bg": "#F0F0F0", "text": "#606060", "border": "#D0D0D0"}
        }

    def _setup_context_menu(self):
        """Enable text selection and copy"""
        self.chat_display.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.chat_display.customContextMenuRequested.connect(self._show_context_menu)
        self.chat_display.setUndoRedoEnabled(False)
        self.chat_display.setReadOnly(True)

    def _show_context_menu(self, pos):
        menu = QMenu()
        copy_action = QAction("Copy", self.chat_display)
        copy_action.triggered.connect(self._copy_selected_text)
        menu.addAction(copy_action)
        menu.exec(self.chat_display.mapToGlobal(pos))

    def _copy_selected_text(self):
        cursor = self.chat_display.textCursor()
        if cursor.hasSelection():
            QGuiApplication.clipboard().setText(cursor.selectedText())

    def add_message(self, sender, message, timestamp):
        """Add formatted message bubble"""
        style = self.styles["user" if sender == "You" else "ai"]
        formatted_content = self._format_content(message)
        html = f"""
        <div style='margin:8px; border-radius:12px; background:{style['bg']};
                    border:1px solid {style['border']}; padding:12px;
                    max-width:80%; float:{'right' if sender == 'You' else 'left'}'>
            <small style='color:{style['text']}; opacity:0.7;'>{timestamp}</small>
            <div style='color:{style['text']}; margin-top:4px; white-space: pre-wrap;'>
                {formatted_content}
            </div>
        </div>
        <div style='clear:both'></div>
        """
        self.chat_display.append(html)
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    def _format_content(self, text):
        """XSS-safe markdown rendering"""
        text = html.escape(text)
        html_content = markdown.markdown(
            text,
            extensions=['fenced_code', 'nl2br'],
            output_format='html5'
        )
        return html_content.replace('<pre><code>', '<pre>').replace('</code></pre>', '</pre>')

class ApiWorker(QThread):
    """Handles API communication with retries"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, session_id, user_message, config_manager, session_manager):
        super().__init__()
        self.session_id = session_id
        self.user_message = user_message
        self.config_manager = config_manager
        self.session_manager = session_manager

    def run(self):
        max_retries = 3
        retry_delay = 2
        ai_message = ""
        
        for attempt in range(max_retries):
            try:
                provider = self.config_manager.get_active_provider()
                config = self.config_manager.get_provider_config(provider)
                history = self.session_manager.sessions[self.session_id]["conversation_history"]
                system_prompt = config.get("system_prompt", "Think of yourself as a great software developer, writer and presenter/toastmaster who understands the nuances of English and Simplified Chinese and can speak convincingly like a native speaker. Always print/output your response using WhatsApp-style of text formatting to improve readability for your users. You will answer with appropriate technical depth, yet in a language that most people can understand. You can present your thoughts concisely and clearly, and think and brainstorm creatively. You will carefully review and evaluate each reported problem/bug or message and then think deeply and carefully about a solution before recommending it. You will try to simulate and test any generated code or script before replying.")
                
                converted_history = self.session_manager.convert_conversation_history(
                    history, provider, system_prompt
                )

                if provider == "Google Gemini":
                    genai.configure(api_key=config["api_key"])
                    model = genai.GenerativeModel(config.get("model", "gemini-2.0-flash-exp"))
                    response = model.generate_content(converted_history)
                    ai_message = response.text

                elif provider == "OpenAI":
                    client = OpenAI(api_key=config["api_key"])
                    response = client.chat.completions.create(
                        model=config.get("model", "gpt-4o-mini"),
                        messages=converted_history,
                        temperature=config.get("temperature", 0.7)
                    )
                    ai_message = response.choices[0].message.content

                elif provider == "Anthropic Claude":
                    client = anthropic.Anthropic(api_key=config["api_key"])
                    response = client.messages.create(
                        system=system_prompt,  # Correct system param
                        messages=converted_history,
                        max_tokens=10000,
                        temperature=config.get("temperature", 0.7)
                    )
                    ai_message = response.content[0].text

                elif provider == "xAI Grok":
                    headers = {
                        "Authorization": f"Bearer {config['api_key']}",
                        "Content-Type": "application/json"
                    }
                    data = {
                        "model": config.get("model", "grok-2-1212"),
                        "messages": converted_history,
                        "temperature": config.get("temperature", 0.7),
                        "max_tokens": 10000,
                    }
                    
                    for retry in range(max_retries):
                        try:
                            response = requests.post(
                                f"{config['base_url']}/chat/completions",
                                headers=headers,
                                json=data
                            )
                            if response.status_code == 429 and retry < max_retries - 1:
                                sleep_time = retry_delay * (2 ** attempt)
                                time.sleep(sleep_time)
                                continue
                                
                            response.raise_for_status()
                            ai_message = response.json()['choices'][0]['message']['content']
                            break
                            
                        except Exception as e:
                            if retry == max_retries - 1:
                                raise

                elif provider == "Ollama":
                    process = subprocess.Popen(
                        ["ollama", "run", config.get("ollama_model", "llama2")],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    full_prompt = "\n".join(
                        [f"{msg['role']}: {msg['content']}" 
                         for msg in converted_history]
                    )
                    stdout, stderr = process.communicate(input=full_prompt)
                    if stderr:
                        raise Exception(stderr)
                    ai_message = stdout.strip()

                elif provider == "OpenAI-Compatible":
                    client = OpenAI(
                        base_url=config["base_url"],
                        api_key=config["api_key"]
                    )
                    response = client.chat.completions.create(
                        model=config["model"],
                        messages=converted_history,
                        temperature=config.get("temperature", 0.7)
                    )
                    ai_message = response.choices[0].message.content

                elif provider == "DeepSeek":
                    headers = {
                        "Authorization": f"Bearer {config['api_key']}",
                        "Content-Type": "application/json"
                    }
                    data = {
                        "model": config["model"],
                        "messages": converted_history,
                        "temperature": config.get("temperature", 0.7)
                    }
                    response = requests.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers=headers,
                        json=data
                    )
                    response.raise_for_status()
                    ai_message = response.json()['choices'][0]['message']['content']

                self.finished.emit(ai_message)
                return

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    self.error.emit(f"API Error ({provider}): {str(e)}")
                    logging.error(f"API call failed: {str(e)}")

class APIConfigDialog(QDialog):
    """API configuration dialog with temperature slider"""
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setWindowTitle("API Configuration")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._setup_ui()
        self._load_current_config()

    def _setup_ui(self):
        layout = QGridLayout()
        self.provider_combo = QComboBox()
        self.provider_combo.addItems([
            "Google Gemini", "OpenAI", "Anthropic Claude", 
            "xAI Grok", "Ollama", "OpenAI-Compatible", "DeepSeek"
        ])
        layout.addWidget(QLabel("Provider:"), 0, 0)
        layout.addWidget(self.provider_combo, 0, 1)

        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("API Key")
        layout.addWidget(QLabel("API Key:"), 1, 0)
        layout.addWidget(self.api_key_input, 1, 1)

        self.model_combo = QComboBox()
        layout.addWidget(QLabel("Model:"), 2, 0)
        layout.addWidget(self.model_combo, 2, 1)

        self.base_url_input = QLineEdit()
        layout.addWidget(QLabel("Base URL:"), 3, 0)
        layout.addWidget(self.base_url_input, 3, 1)

        self.system_prompt_input = QTextEdit()
        layout.addWidget(QLabel("System Prompt:"), 4, 0)
        layout.addWidget(self.system_prompt_input, 4, 1)

        # Temperature slider
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_label = QLabel("Temperature: 0.7")
        layout.addWidget(QLabel("Temperature:"), 5, 0)
        layout.addWidget(self.temp_slider, 5, 1)
        layout.addWidget(self.temp_label, 5, 2)
        self.temp_slider.valueChanged.connect(self._update_temp_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save_config)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons, 6, 0, 1, 2)

        self.provider_combo.currentTextChanged.connect(self._update_fields)
        self.setLayout(layout)

    def _load_current_config(self):
        provider = self.config_manager.get_active_provider()
        self.provider_combo.setCurrentText(provider)
        self._update_fields(provider)

    def _update_fields(self, provider):
        config = self.config_manager.get_provider_config(provider)
        self.api_key_input.setText(config.get("api_key", ""))
        self.base_url_input.setText(config.get("base_url", ""))
        self.system_prompt_input.setPlainText(config.get("system_prompt", ""))
        
        # Temperature handling
        temp_value = int(config.get("temperature", 0.7) * 100)
        self.temp_slider.setValue(temp_value)
        self._update_temp_label(temp_value)
        
        self.model_combo.clear()
        if provider == "Google Gemini":
            self.model_combo.addItems(["gemini-2.0-flash-exp", "gemini-pro", "gemini-1.5-flash"])
        elif provider == "OpenAI":
            self.model_combo.addItems(["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
        elif provider == "Anthropic Claude":
            self.model_combo.addItems(["claude-3-opus-20240229", "claude-3-sonnet-20240229"])
        elif provider == "xAI Grok":
            self.model_combo.addItems(["grok-2-1212", "grok-beta", "grok-2-latest", "grok-2-mini"])
        elif provider == "Ollama":
            self._load_ollama_models()
        elif provider == "OpenAI-Compatible":
            self.model_combo.addItems(["deepseek-ai/DeepSeek-V3", "custom-model", "llama2-13b-chat", "mixtral-8x7b"])
            self.base_url_input.setVisible(True)
        elif provider == "DeepSeek":
            self.model_combo.addItems(["deepseek-chat", "deepseek-coder"])
            self.base_url_input.setVisible(False)

    def _update_temp_label(self, value):
        self.temp_label.setText(f"Temperature: {value/100:.1f}")

    def _load_ollama_models(self):
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            models = [line.split()[0] for line in result.stdout.splitlines()[1:]]
            self.model_combo.addItems(models)
        except Exception as e:
            QMessageBox.warning(self, "Ollama Error", f"Failed to load models: {str(e)}")

    def _save_config(self):
        provider = self.provider_combo.currentText()
        
        # Validate Grok URL
        if provider == "xAI Grok":
            base_url = self.base_url_input.text().strip()
            if not base_url.startswith("https://api.x.ai"):
                QMessageBox.critical(self, "Error", 
                    "Grok Base URL must point to https://api.x.ai")
                return
                
        config = {
            "api_key": self.api_key_input.text(),
            "base_url": self.base_url_input.text(),
            "model": self.model_combo.currentText(),
            "system_prompt": self.system_prompt_input.toPlainText(),
            "temperature": round(self.temp_slider.value() / 100.0, 1)
        }
        if provider == "OpenAI" and not config["model"].startswith("gpt-4"):
            QMessageBox.warning(self, "Model Warning", 
                "GPT-4 models recommended for best results")
        if provider == "Ollama":
            config["ollama_model"] = self.model_combo.currentText()
        elif provider == "DeepSeek":
            if not config["api_key"]:
                QMessageBox.critical(self, "Error", "API Key is required for DeepSeek")
                return
        
        self.config_manager.config[provider] = config
        self.config_manager.set_active_provider(provider)
        self.config_manager.save_config()
        self.accept()

class ModernChatWindow(QMainWindow):
    """Main application window with modern UI"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Chat Studio")
        self.setGeometry(100, 100, 1200, 800)
        self._setup_menubar()
        self._setup_ui()
        self._initialize_managers()
        self._connect_signals()
        self._load_initial_session()
        self._apply_styles()

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F0F2F5;
                font-family: 'Segoe UI', Arial;
            }
            QListWidget {
                background: white;
                border-radius: 10px;
                padding: 8px;
            }
            QListWidget::item {
                padding: 12px;
                border-bottom: 1px solid #E0E0E0;
            }
            QListWidget::item:selected {
                background: #E3F2FD;
            }
            QPushButton {
                background: #0084FF;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 20px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #0073E6;
            }
            QTextEdit, QTextBrowser {
                background: white;
                border: 1px solid #E0E0E0;
                border-radius: 15px;
                padding: 15px;
                font-size: 14px;
            }
            QLineEdit {
                border: 2px solid #E0E0E0;
                border-radius: 20px;
                padding: 10px 15px;
                font-size: 14px;
            }
            QSlider::groove:horizontal {
                background: #E0E0E0;
                height: 4px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #0084FF;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
        """)

    def _setup_menubar(self):
        """Proper menu bar implementation"""
        menubar = QMenuBar(self)
        
        # File menu
        file_menu = menubar.addMenu("&File")
        new_action = QAction("&New Chat", self)
        new_action.triggered.connect(self._create_new_chat)
        file_menu.addAction(new_action)
        
        export_action = QAction("&Export Chat...", self)
        export_action.triggered.connect(self._export_chat)
        file_menu.addAction(export_action)
        
        # Settings menu
        settings_menu = menubar.addMenu("&Settings")
        config_action = QAction("&API Configuration...", self)
        config_action.triggered.connect(self._show_api_config)
        settings_menu.addAction(config_action)
        
        self.setMenuBar(menubar)

    def _setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # Left panel (Chat list)
        left_panel = QWidget()
        left_panel.setFixedWidth(280)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.chat_list = QListWidget()
        self.chat_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_layout.addWidget(self.chat_list)
        
        # Chat management buttons
        self.new_btn = QPushButton("New Chat")
        self.delete_btn = QPushButton("Delete Chat")
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.new_btn)
        button_layout.addWidget(self.delete_btn)
        left_layout.addLayout(button_layout)
        main_layout.addWidget(left_panel)

        # Right panel (Chat display)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.chat_display = QTextBrowser()
        self.chat_display.setSizePolicy(
            QSizePolicy.Policy.Expanding, 
            QSizePolicy.Policy.Expanding
        )
        right_layout.addWidget(self.chat_display, 1)  # Add stretch factor
        
        # Input area
        input_panel = QWidget()
        input_layout = QVBoxLayout(input_panel)
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Type your message...")
        self.input_field.setMaximumHeight(100)
        input_layout.addWidget(self.input_field)
        
        # Toolbar with emoji button
        toolbar = QHBoxLayout()
        self.attach_btn = QPushButton("Attach File")
        self.emoji_btn = QPushButton("😊 Emoji")
        self.send_btn = QPushButton("Send")
        toolbar.addWidget(self.attach_btn)
        toolbar.addWidget(self.emoji_btn)
        toolbar.addWidget(self.send_btn)
        input_layout.addLayout(toolbar)
        right_layout.addWidget(input_panel)
        
        main_layout.addWidget(right_panel, 1)

    def _initialize_managers(self):
        self.config_manager = APIConfigManager()
        self.session_manager = ChatSessionManager()
        self.message_manager = ModernMessageManager(self.chat_display)
        self.current_session_id = None

    def _connect_signals(self):
        self.new_btn.clicked.connect(self._create_new_chat)
        self.delete_btn.clicked.connect(self._delete_chat)
        self.send_btn.clicked.connect(self._send_message)
        self.attach_btn.clicked.connect(self._attach_file)
        self.emoji_btn.clicked.connect(self._show_emoji_picker)
        self.chat_list.itemClicked.connect(self._load_selected_chat)

    def _load_initial_session(self):
        if self.session_manager.sessions:
            sessions = sorted(
                self.session_manager.sessions.items(),
                key=lambda x: x[1]["created_at"],
                reverse=True
            )
            self.current_session_id = sessions[0][0]
            self._update_chat_list()
            self._update_chat_display()

    def _create_new_chat(self):
        session_id, session_name = self.session_manager.new_session()
        self.current_session_id = session_id
        self._update_chat_list()
        self._update_chat_display()

    def _update_chat_list(self):
        self.chat_list.clear()
        sessions = sorted(
            self.session_manager.sessions.items(),
            key=lambda x: x[1]["created_at"],
            reverse=True
        )
        for session_id, data in sessions:
            item = QListWidgetItem(data["session_name"])
            item.setData(Qt.ItemDataRole.UserRole, session_id)
            self.chat_list.addItem(item)

    def _update_chat_display(self):
        self.chat_display.clear()
        if self.current_session_id in self.session_manager.sessions:
            session = self.session_manager.sessions[self.current_session_id]
            for sender, message, timestamp in session["chat_log"]:
                self.message_manager.add_message(sender, message, timestamp)
            
            if session["attached_files"]:
                self.chat_display.append("<div style='color:#666; margin:16px;'>📎 Attached Files:")
                for file_name in session["attached_files"]:
                    self.chat_display.append(f"• {file_name}</div>")

    def _send_message(self):
        self.send_btn.setEnabled(False)  # Prevent duplicate sends
        QApplication.processEvents()  # Keep UI responsive
        message = self.input_field.toPlainText().strip()
        if not message:
            self.send_btn.setEnabled(True)
            return
        
        if not self.current_session_id:
            self._create_new_chat()
            
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        self.session_manager.add_message(
            self.current_session_id, "You", message, timestamp
        )
        self.message_manager.add_message("You", message, timestamp)
        
        self.worker = ApiWorker(
            self.current_session_id,
            message,
            self.config_manager,
            self.session_manager
        )
        self.worker.finished.connect(self._handle_response)
        self.worker.error.connect(self._handle_error)
        self.worker.start()
        
        self.input_field.clear()

    def _handle_response(self, response):
        self.send_btn.setEnabled(True)
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        self.session_manager.add_message(
            self.current_session_id, "AI", response, timestamp
        )
        self.message_manager.add_message("AI", response, timestamp)
        self.session_manager.save_session(self.current_session_id)

    def _handle_error(self, error):
        self.send_btn.setEnabled(True)
        QMessageBox.critical(self, "API Error", error)

    def _attach_file(self):
        file_path, _ = QFileDialog.getOpenFileName()
        if file_path:
            try:
                file_name = self.session_manager.attach_file(
                    self.current_session_id, file_path
                )
                timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                self.session_manager.add_message(
                    self.current_session_id, "System", f"Attached: {file_name}", timestamp
                )
                self._update_chat_display()
            except Exception as e:
                QMessageBox.critical(self, "Attachment Error", str(e))

    def _load_selected_chat(self, item):
        session_id = item.data(Qt.ItemDataRole.UserRole)
        self.current_session_id = session_id
        self._update_chat_display()

    def _delete_chat(self):
        """Added error handling for file deletion"""
        current_item = self.chat_list.currentItem()
        if current_item:
            session_id = current_item.data(Qt.ItemDataRole.UserRole)
            try:
                self.session_manager.delete_session(session_id)
            except Exception as e:
                QMessageBox.critical(self, "Deletion Error", str(e))
            self._update_chat_list()
            if self.current_session_id == session_id:
                self._create_new_chat()

    def _export_chat(self):
        if not self.current_session_id:
            QMessageBox.warning(self, "Error", "No active chat to export")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Chat", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                session_data = self.session_manager.sessions[self.current_session_id]
                with open(file_path, 'w') as f:
                    json.dump(session_data, f, indent=4)
                QMessageBox.information(self, "Success", "Chat exported successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def _show_emoji_picker(self):
        """Modern emoji picker implementation"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Emoji")
        layout = QGridLayout()
        emojis = ["👍", "🙏", "😊", "😅", "😇", "😔", "😢", "😄", "🥳", "😘", "🤗", "😂", "🍻", "🤩", "👏", "👌", "✌️", "☝️", "👎", "👋", "💪", "🫶", "🥱", "😴", "🙄", "🤡", "💩", "😭", "😤", "😡", "☹️", "😣", "😖", "😫", "😎", "🤓", "🧐", "🤪", "😜", "😝", "😍", "🥰", "😚", "😋", "😀", "😃", "😄", "😁", "😆", "😅", "😂", "🤣", "😊", "😇",
                 "😍", "🤩", "😘", "😗", "😚", "😋", "😛", "😝", "🤑", "🤗"]
        
        for i, emoji in enumerate(emojis):
            btn = QPushButton(emoji)
            btn.clicked.connect(lambda _, e=emoji: self._insert_emoji(e))
            layout.addWidget(btn, i//5, i%5)
            
        dialog.setLayout(layout)
        dialog.exec()

    def _insert_emoji(self, emoji):
        """Insert emoji at current cursor position"""
        cursor = self.input_field.textCursor()
        cursor.insertText(emoji)
        self.input_field.setFocus()

    def _show_api_config(self):
        dialog = APIConfigDialog(self.config_manager, self)
        dialog.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModernChatWindow()
    window.show()
    sys.exit(app.exec())

