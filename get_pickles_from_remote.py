import os
import time
import subprocess
import paramiko

# --- 설정 정보 ---
REMOTE_HOST = "185.65.93.114"
REMOTE_PORT = "45474"
REMOTE_USER = "root"
KEY_PATH = "/home/bongo/porter_notebook/research/research.pem"

REMOTE_DIR = "/root/qwen3_AG/AttentionHeads_exo/" 
LOCAL_DIR = "/home/bongo/porter_notebook/research/qwen3/AttentionHeads_exo/output_results2/"
INTERVAL = 10 

# search_pattern = f"{REMOTE_DIR}exo_attention_result_32B_2_*.pkl"
search_pattern = f"{REMOTE_DIR}exo_sample_32B_*.pkl"

def get_and_delete_large_files():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # 1. 파일 목록 확인을 위해 SSH 접속
        key = paramiko.RSAKey.from_private_key_file(KEY_PATH)
        ssh.connect(REMOTE_HOST, port=int(REMOTE_PORT), username=REMOTE_USER, pkey=key)
        
        # 특정 패턴 파일 목록 가져오기
        
        stdin, stdout, stderr = ssh.exec_command(f"ls {search_pattern}")
        file_list = stdout.read().decode().split()
        
        if not file_list:
            print(f"[{time.strftime('%H:%M:%S')}] 대상 파일 없음.")
            return

        for remote_file_path in file_list:
            file_name = os.path.basename(remote_file_path)
            local_file_path = os.path.join(LOCAL_DIR, file_name)
            
            print(f"[{time.strftime('%H:%M:%S')}] {file_name} (2GB) 전송 시작...")
            
            # 2. 시스템 SCP 실행 (subprocess)
            # -q: 진행바 숨기기 (필요하면 빼셔도 됩니다)
            scp_cmd = [
                "scp", "-P", REMOTE_PORT,
                "-i", KEY_PATH,
                "-o", "StrictHostKeyChecking=no", # 처음 접속 시 yes/no 묻는 것 방지
                f"{REMOTE_USER}@{REMOTE_HOST}:{remote_file_path}",
                local_file_path
            ]
            
            # subprocess.run은 프로세스가 종료될 때까지 대기합니다.
            result = subprocess.run(scp_cmd)
            
            # 3. 리턴 코드가 0이면 성공적으로 완료된 것
            if result.returncode == 0:
                print(f"성공: {file_name} 전송 완료. 원격 파일 삭제 중...")
                ssh.exec_command(f"rm {remote_file_path}")
            else:
                print(f"실패: {file_name} 전송 중 오류 발생. 삭제하지 않습니다.")

    except Exception as e:
        print(f"에러 발생: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)
        
    print(f"대용량 파일(2GB) 모니터링 및 즉시 삭제 루프 시작...")
    while True:
        get_and_delete_large_files()
        time.sleep(INTERVAL)