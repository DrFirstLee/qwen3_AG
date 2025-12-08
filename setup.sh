#!/bin/bash
set -e

# tmux 자동 실행 방지
touch ~/.no_auto_tmux

# 기본 패키지 업데이트 및 설치
sudo apt update -y
sudo apt install -y libgl1-mesa-glx nano zip wget

# Anaconda 자동 설치
ANACONDA_SH=Anaconda3-2024.02-1-Linux-x86_64.sh
wget https://repo.anaconda.com/archive/$ANACONDA_SH
bash $ANACONDA_SH -b -p $HOME/anaconda3

# PATH 설정
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

conda init



# Conda 버전 확인
conda --version

# gdown 설치 및 데이터 다운로드
pip install --upgrade pip && pip install gdown
gdown 1OEz25-u1uqKfeuyCqy7hmiOv7lIWfigk

# 데이터 압축 해제
unzip AGD20K.zip


# 깃허브 레포 클론
git clone git@github.com:DrFirstLee/qwen3_AG.git
git clone https://github.com/DrFirstLee/qwen3_AG.git



# qwen25 환경 생성
cd qwen_AG_new
conda env create -f qwen3.yaml -n qwen3

# 생성된 환경 활성화 안내
echo ">>> qwen3 환경 생성 완료. 다음 명령으로 활성화하세요:"
echo "conda activate qwen3"


ssh -p 40743 root@185.65.93.114 -L 8080:localhost:8080

ssh -i /home/bongo/porter_notebook/research/research.pem -p 40743 root@185.65.93.114  "tar -C /root/qwen3_AG/APM_dot_verifying/ -czf - top_attention_heads" | tar -xzvf - -C /home/bongo/porter_notebook/research/qwen3/
ssh -i /home/bongo/porter_notebook/research/research.pem -p 40743 root@185.65.93.114  "tar -C /root/qwen3_AG/APM_dot_verifying -czf - 32B_exo_random2" | tar -xzvf - -C /home/bongo/porter_notebook/research/qwen3/

/root/anaconda3/envs/qwen3/bin/python ego_only_relative.py
/venv/qwen3/bin/python ego_only_relative.py


(qwen3) (main) root@C.28152234:~/qwen3_AG$