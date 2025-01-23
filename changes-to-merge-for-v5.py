import sys
import os
import json
import yaml
import logging
import time
import fitz
import io
import pythoncom
from datetime import datetime
from PIL import Image
from docx import Document
import pandas as pd
import google.generativeai as genai
from openai import OpenAI
import anthropic
import requests
import subprocess
import html
import markdown
import hashlib
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QPushButton, QTextEdit, QTextBrowser, QFileDialog, QDialog,
    QGridLayout, QMessageBox, QLineEdit, QComboBox, QSlider, QMenuBar,
    QScrollArea, QDialogButtonBox, QSizePolicy, QMenu, QListWidgetItem, QDialog,
    QStatusBar, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDir, QSize, QTimer
from PyQt6.QtGui import (
    QFont, QKeySequence, QTextCursor, QColor, QAction, QIcon,
    QTextCharFormat, QPalette, QGuiApplication, QPixmap, QImage
)
from transformers import AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration
import torch

# Configure logging
logging.basicConfig(filename='mychat.log', level=logging.ERROR,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure directories exist
for dir in ["chat_logs", "config"]:
    if not os.path.exists(dir):
        os.makedirs(dir)

class FileProcessingThread(QThread):
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)

    def __init__(self, processor, file_path):
        super().__init__()
        self.processor = processor
        self.file_path = file_path

    def run(self):
        try:
            if self.file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                result = self.processor.process_image(self.file_path)
            elif self.file_path.lower().endswith('.pdf'):
                result = self.processor.process_pdf(self.file_path)
            elif self.file_path.lower().endswith(('.docx', '.xlsx')):
                result = self.processor.process_office(self.file_path)
            self.finished.emit(self.file_path, result)
        except Exception as e:
            self.error.emit(str(e))

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

    # ... (Keep existing APIConfigManager methods unchanged) ...

class ChatSessionManager:
    """Manages chat sessions with file persistence"""
    def __init__(self):
        self.sessions = {}
        self.load_sessions()

    # ... (Keep existing ChatSessionManager methods unchanged) ...

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

    # ... (Keep existing ModernMessageManager methods unchanged) ...

class ApiWorker(QThread):
    """Handles API communication with retries"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    # ... (Keep existing ApiWorker implementation unchanged) ...

class APIConfigDialog(QDialog):
    """API configuration dialog with temperature slider"""
    # ... (Keep existing APIConfigDialog implementation unchanged) ...

class ModernChatWindow(QMainWindow):
    """Main application window with modern UI and multimodal support"""
    def __init__(self):
        super().__init__()
        self.settings = QSettings("MultiAI", "ChatApp")
        self.setWindowTitle("AI Chat Studio - Multimodal Edition")
        self.setGeometry(100, 100, 1400, 900)
        self._setup_menubar()
        self._setup_ui()
        self._initialize_managers()
        self._connect_signals()
        self._load_initial_session()
        self._apply_styles()
        self.init_models()
        self.temp_files = []
        self.load_settings()

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
            #previewLabel {
                background: white;
                border-radius: 12px;
                padding: 10px;
            }
        """)

    def init_models(self):
        """Initialize all AI models"""
        try:
            # Vision Model
            self.image_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            self.image_model = LlavaForConditionalGeneration.from_pretrained(
                "llava-hf/llava-1.5-7b-hf", 
                device_map="auto",
                load_in_4bit=True,
                torch_dtype=torch.float16
            )
            
            # Document Model
            self.layout_tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")

            self.statusBar().showMessage("Models loaded successfully", 3000)
        except Exception as e:
            self._show_error_message("Model Error", str(e))

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
        
        self.chat_list = QListWidget()
        left_layout.addWidget(self.chat_list)
        
        # Chat management buttons
        self.new_btn = QPushButton("New Chat")
        self.delete_btn = QPushButton("Delete Chat")
        left_layout.addWidget(self.new_btn)
        left_layout.addWidget(self.delete_btn)
        main_layout.addWidget(left_panel)

        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Chat display area
        self.chat_display = QTextBrowser()
        right_layout.addWidget(self.chat_display, 4)

        # Preview area
        self.preview_label = QLabel()
        self.preview_label.setObjectName("previewLabel")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll = QScrollArea()
        scroll.setWidget(self.preview_label)
        right_layout.addWidget(scroll, 1)

        # Input area
        input_panel = QWidget()
        input_layout = QVBoxLayout(input_panel)
        
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Type your message or upload a file...")
        input_layout.addWidget(self.input_field)
        
        # Toolbar
        toolbar = QHBoxLayout()
        self.attach_btn = QPushButton("📁 Upload")
        self.emoji_btn = QPushButton("😊 Emoji")
        self.send_btn = QPushButton("Send")
        toolbar.addWidget(self.attach_btn)
        toolbar.addWidget(self.emoji_btn)
        toolbar.addWidget(self.send_btn)
        input_layout.addLayout(toolbar)
        
        right_layout.addWidget(input_panel)
        main_layout.addWidget(right_panel)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)

    def _initialize_managers(self):
        self.config_manager = APIConfigManager()
        self.session_manager = ChatSessionManager()
        self.message_manager = ModernMessageManager(self.chat_display)
        self.current_session_id = None

    def _connect_signals(self):
        self.new_btn.clicked.connect(self._create_new_chat)
        self.delete_btn.clicked.connect(self._delete_chat)
        self.send_btn.clicked.connect(self._send_message)
        self.attach_btn.clicked.connect(self._handle_file_upload)
        self.emoji_btn.clicked.connect(self._show_emoji_picker)
        self.chat_list.itemClicked.connect(self._load_selected_chat)

    def _handle_file_upload(self):
        """Handle file uploads with preview"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "",
            "Supported Files (*.png *.jpg *.jpeg *.pdf *.docx *.xlsx);;All Files (*)"
        )

        if file_path:
            self.progress_bar.setValue(0)
            self.preview_label.clear()
            
            # Show preview
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                self._show_image_preview(file_path)
            else:
                self.preview_label.setText(f"Preview: {os.path.basename(file_path)}")

            # Start processing
            self.worker = FileProcessingThread(self, file_path)
            self.worker.finished.connect(self._handle_processed_file)
            self.worker.error.connect(self._handle_processing_error)
            self.worker.start()

    def process_image(self, file_path):
        """Process image files using LLaVA"""
        try:
            image = Image.open(file_path).convert("RGB")
            prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
            
            inputs = self.image_processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.image_model.device)

            generated_ids = self.image_model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.1
            )
            
            description = self.image_processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0].replace(prompt, "").strip()
            
            return description
        except Exception as e:
            raise RuntimeError(f"Image processing failed: {str(e)}")

    def process_pdf(self, file_path):
        """Process PDF files with layout analysis"""
        try:
            doc = fitz.open(file_path)
            text = []
            for page in doc:
                blocks = page.get_text("blocks")
                for b in blocks:
                    text.append(f"[Page {page.number} @ ({b[0]:.1f},{b[1]:.1f})]: {b[4]}")
            return "\n".join(text)[:5000] + "..."
        except Exception as e:
            raise RuntimeError(f"PDF processing failed: {str(e)}")

    def process_office(self, file_path):
        """Process Office documents"""
        try:
            if file_path.endswith('.docx'):
                doc = Document(file_path)
                return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
                return df.to_markdown(index=False)
        except Exception as e:
            raise RuntimeError(f"Office doc processing failed: {str(e)}")

    def _show_image_preview(self, file_path):
        """Display image preview with proper scaling"""
        pixmap = QPixmap(file_path)
        if pixmap.width() > 800 or pixmap.height() > 600:
            pixmap = pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio)
        self.preview_label.setPixmap(pixmap)

    def _handle_processed_file(self, file_path, result):
        """Inject processed file content into chat"""
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        file_name = os.path.basename(file_path)
        
        # Add to session
        self.session_manager.add_message(
            self.current_session_id, "System", 
            f"File Analysis ({file_name}):\n{result}", 
            timestamp
        )
        
        # Update display
        self.message_manager.add_message("System", f"File Analysis ({file_name}):\n{result}", timestamp)
        self.session_manager.save_session(self.current_session_id)
        self.progress_bar.setValue(100)

    def _handle_processing_error(self, error):
        self.progress_bar.setValue(0)
        self._show_error_message("Processing Error", error)

    # ... (Keep remaining methods from original v4 implementation unchanged) ...

    def closeEvent(self, event):
        """Cleanup on exit"""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        if hasattr(self, 'image_model'):
            del self.image_model
        event.accept()

if __name__ == "__main__":
    pythoncom.CoInitialize()
    app = QApplication(sys.argv)
    window = ModernChatWindow()
    window.show()
    sys.exit(app.exec())
