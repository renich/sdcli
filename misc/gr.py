import gradio as gr

demo = gr.load("Helsinki-NLP/opus-mt-en-es", src="models")

demo.launch()
