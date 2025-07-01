"""import numpy as np
import gradio as gr
import os
import json
import datetime

import app_theme


class ControlPanel:
    def __init__(self):


        self.init_ui()
        self.app.queue()
        self.app.launch(max_threads=8, server_port=5000, share=True)


    def change_time_scale(self, time_scale=1):


    def init_ui(self):
        with gr.Blocks(title="ML Agents", theme=app_theme.Softy()) as self.app:
            gr.Markdown("## Agents Control Panel")

            # Guess tab
            with gr.Tab("Unity"):
                with gr.Row():
                    with gr.Accordion():
                        set_timescale_button = gr.Button(value="Set Timescale", variant="secondary")
                        set_timescale_button.click(fn=load_image, inputs=None, outputs=[display_image], every=1)
                        whisper_button = gr.Button(value="refresh text", variant="secondary")
"""
