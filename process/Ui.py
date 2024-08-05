import gradio as gr
import cropvideo
def crop_video(inputFile):
    return cropvideo.process_video(inputFile)

if __name__=="__main__":
    with gr.Blocks(analytics_enabled=False) as synctalk_process:
         gr.Markdown("<div align='视频预处理'> <h2> video_process: </span> </h2> </div>")
         with gr.Row():
             with gr.Column(variant='panel'):
                 with gr.TabItem('upload video'):
                     with gr.Column(variant='panel'):
                         #本地上传要预处理的原视频
                         video_client=gr.Video(sources="upload",label='upload video',format='mp4',interactive=True)
                         vc_button=gr.Button("本地crop_video") 
                 with gr.Row():       
                    with gr.TabItem('server video'):
                        with gr.Column(variant='panel'):
                            #服务器上传
                            video_server=gr.FileExplorer(glob="**/*.mp4",value='',file_count='single',root_dir='/home/weiyubin/process/video/',label='load .mp4 file to train',interactive=True)
                            vs_button=gr.Button("服务器crop_video")         
             with gr.Column(variant='panel'):
                 gen_video = gr.Video(label="Generated video", format="mp4", visible=True)
                 vs_button.click(crop_video,inputs=[video_server],outputs=[gen_video],queue=True)
                 #输出的视频路径与输入在同一个路径下/home/weiyubin/process/video/，名字为
                 vc_button.click(crop_video,inputs=[video_client],outputs=[gen_video],queue=True)
                 
    synctalk_process.launch(debug=True)
        
        
 