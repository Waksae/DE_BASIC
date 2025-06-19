# DE_BASIC

# 두 번째 튜토리얼
- fastapi 컨테이너를 띄우고
- streamlit 컨테이너 띄우고
- 위 두 컨테이너를 동시에 관리

# mysql 활성화 코드 순서
sudo apt update
sudo apt install mysql-server
sudo ufw all mysql
sudo systemctl start mysql
sudo systemctl enable mysql

sudo systemctl status mysql