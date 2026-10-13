"""Generate three model-specific MD/notebooks from one shared reviewed template."""
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODELS={
 'xception':{'TITLE':'FFPP_c23_Xception_Baseline_Seed42','SCOPE':'인코더 전체 + 분류기 fine-tuning','ENCODER_LR':'1e-4','TRANSFORM':'bicubic299 direct square resize, mean/std0.5 정규화'},
 'clip_b16':{'TITLE':'FFPP_c23_CLIP_ViTB16_Baseline_Seed42','SCOPE':'마지막2개 Transformer 블록 + 최종 LayerNorm·visual projection + 분류기 fine-tuning. 나머지 인코더 동결','ENCODER_LR':'1e-5','TRANSFORM':'OpenAI CLIP bicubic224/center crop + CLIP mean/std 정규화. 512차원 visual feature에 Linear512→2, text encoder/추가 L2 normalize 없음'},
 'dinov2_b14':{'TITLE':'FFPP_c23_DINOv2_ViTB14_Baseline_Seed42','SCOPE':'마지막2개 Transformer 블록 + 최종 LayerNorm + 분류기 fine-tuning. 나머지 인코더 동결','ENCODER_LR':'1e-5','TRANSFORM':'bicubic224 direct square resize + ImageNet mean/std 정규화. CLS768 특징에 Linear768→2. 추가 center crop 없음'}}


def main():
    template=(ROOT/'docs/colab/_ffpp_c23_baseline_template.md').read_text()
    stems=[]
    for model,settings in MODELS.items():
        text=template
        for key,value in {'MODEL':model,**settings}.items():text=text.replace('{{'+key+'}}',value)
        if '{{' in text:raise ValueError('Template placeholder unresolved')
        stem=f'train_ffpp_c23_{model}_seed42';stems.append(stem)
        (ROOT/'docs/colab'/f'{stem}.md').write_text(text)
    subprocess.run([sys.executable,str(ROOT/'scripts/build_celebdf_training_notebooks.py'),*stems],check=True)


if __name__=='__main__':main()
