# Python 3.11 슬림 버전 사용 (용량 작음)
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치 (필요한 것만)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 먼저 설치 (캐싱 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사
COPY . .

# 데이터 디렉토리 생성
RUN mkdir -p /app/data/database /app/data/raw

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    #compose file 기본 경로 설정
    COMPOSE_FILE=docker/docker-compose.yml 

# 기본 명령어 (필요시 override)
CMD ["python", "--version"]