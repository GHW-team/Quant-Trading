# Quant-Trading


# 🚀 머신러닝 퀀트 트레이딩 시스템

## 팀원
- 경제 전문가: [이름]
- 데이터 사이언티스트: [이름]
- 머신러닝 엔지니어: [이름]

## 프로젝트 설명
절대 모멘텀 전략 기반 한국 주식 자동매매 시스템

## 🛠️ 개발 환경 설정

### 1. 사전 준비
- Docker Desktop 설치 (https://www.docker.com/products/docker-desktop)
- Git 설치
- GitHub 계정

### 2. 프로젝트 클론
\`\`\`bash
git clone https://github.com/your-team/quant_project.git
cd quant_project
\`\`\`

### 3. 환경 변수 설정
\`\`\`bash
cp .env.example .env
# .env 파일을 열어서 본인의 API 키 입력
\`\`\`

### 4. Docker 실행
\`\`\`bash
# 이미지 빌드 및 실행
docker-compose up --build

# 백그라운드 실행
docker-compose up -d

# 특정 서비스만 실행
docker-compose up jupyter
\`\`\`

### 5. Jupyter 접속
브라우저에서 http://localhost:8888 접속

## 📦 주요 명령어

\`\`\`bash
# 컨테이너 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs -f

# 컨테이너 중지
docker-compose down

# 데이터 수집 실행
docker-compose run data-collector python scripts/collect_data.py

# 테스트 실행
docker-compose run data-collector pytest tests/
\`\`\`

## 📂 디렉토리 구조
@
Quant-Trading/
	README.md
	.gitignore
	.env.example
	requirements.txt
	setup.py
	docker/
	config/
	data/
		raw/
		processed/
		features/
		database/
	src/
		data/
		models/
		backtest/
		execution/
	scripts/
	notebooks/
	tests/
	logs/
	models/
	reports/
	docs/[위에서 만든 구조 붙여넣기]

## 🤝 기여 방법
1. 브랜치 생성: \`git checkout -b feature/기능명/사용자명\`
2. 작업 후 커밋: \`git commit -am "설명"\`
3. 푸시: \`git push origin feature/기능명/사용자명\`
4. Pull Request 생성