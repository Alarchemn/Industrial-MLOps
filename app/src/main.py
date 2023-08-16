import gradio as gr
import random
from models.utils import unique_product_id, unique_type, predict

with gr.Blocks() as app:
    gr.Markdown(
        '''**Machine Failure Classification APP**''')
    
    with gr.Row():
        with gr.Column():
            product_id = gr.Dropdown(
                label='Product ID',
                info='Machine ID',
                choices=unique_product_id,
                value=lambda: random.choice(unique_product_id),
                interactive=True
            )
            type_ = gr.Dropdown(
                label='Type',
                info='Type of the machine',
                choices=unique_type,
                value=lambda: random.choice(unique_type),
                interactive=True
            )
            air_temperature = gr.Slider(
                label='Air Temperature',
                info='Kelvin',
                minimum=295,
                maximum=305,
                step=0.01,
                randomize=True,
                interactive=True
            )
            process_temperature = gr.Slider(
                label='Process Temperature',
                info='Kelvin',
                minimum=305,
                maximum=314,
                step=0.01,
                randomize=True,
                interactive=True
            )
            Rotational_speed = gr.Slider(
                label='Rotational Speed',
                info='RPM',
                minimum=1180,
                maximum=2886,
                step=1,
                randomize=True,
                interactive=True
            )
            torque = gr.Slider(
                label='Torque',
                info='Nm',
                minimum=3.7,
                maximum=77,
                step=0.1,
                randomize=True,
                interactive=True
            )
            tool_wear = gr.Slider(
                label='Tool Wear',
                info='min',
                minimum=0,
                maximum=255,
                step=1,
                randomize=True,
                interactive=True
            )
            twf = gr.Checkbox(
                label='TWF',
            )
            hdf = gr.Checkbox(
                label='HDF',
            )
            pwf = gr.Checkbox(
                label='PWF',
            )
            osf = gr.Checkbox(
                label='OSF',
            )
            rnf = gr.Checkbox(
                label='RNF',
            )
        
        with gr.Column():
            label = gr.Label()
            with gr.Row():
                predict_btn = gr.Button(value="Predict")
            predict_btn.click(
                predict,
                inputs=[
                    product_id,
                    type_,
                    air_temperature,
                    process_temperature,
                    Rotational_speed,
                    torque,
                    tool_wear,
                    twf,
                    hdf,
                    pwf,
                    osf,
                    rnf
                ],
                outputs=[label]
            )

if __name__ == '__main__':
    app.launch(server_port=8000, server_name='127.0.0.1')