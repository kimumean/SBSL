# CV Calibration & Scan-Rate Analyzer

VersaStudio에서 생성한 텍스트 기반 `.par` 파일을 브라우저에서 분석하는 Streamlit 웹 애플리케이션입니다. 사용자는 Windows PC에 Python이나 별도 데스크톱 프로그램을 설치할 필요 없이 배포된 URL을 Chrome 또는 Edge로 열어 사용할 수 있습니다.

주요 기능:

- concentration series와 scan-rate series 분석
- `Scan Rate (V/s)` 메타데이터 자동 파싱 및 수동 수정
- 농도의 명시적 수동 입력(파일명이나 전류에서 자동 확정하지 않음)
- segment 및 potential turning direction 기반 cycle/sweep 판별
- anodic/cathodic peak, fixed-potential 보간, interval mean, integrated charge
- 기본 OLS calibration 및 scan-rate 회귀
- Plotly CV overlay 및 calibration/scan-rate plot
- UTF-8-SIG CSV, Plotly HTML, PNG/SVG, 통합 ZIP 다운로드
- 선택적 환경변수/Streamlit secrets 비밀번호 인증
- 세션당 30개, 파일당 20 MB, 전체 200 MB 제한

## 프로젝트 구조

```text
cv_web_analyzer/
├── app.py
├── core/
│   ├── par_parser.py
│   ├── cv_analysis.py
│   ├── regression.py
│   ├── cycle_detection.py
│   └── exporters.py
├── models/
│   └── data_models.py
├── tests/
│   ├── test_par_parser.py
│   ├── test_cycle_detection.py
│   ├── test_peak_analysis.py
│   └── test_regression.py
├── requirements.txt
├── pytest.ini
├── README.md
├── Dockerfile
├── .gitignore
└── .streamlit/
    └── config.toml
```

## `.par` 파싱 방식

이 앱은 `.par`을 일반 CSV로 읽지 않습니다. 파일 전체를 먼저 UTF-8-SIG, UTF-8, CP949, Latin-1 순서로 안전하게 decode하고, 전역 `key=value` 메타데이터와 `<SegmentN> ... </SegmentN>` 블록을 분리합니다. 각 블록의 `Definition=` 행을 해당 블록의 열 정의로 사용한 뒤 데이터 행만 CSV 규칙으로 처리합니다.

내부 표준 열은 다음과 같습니다.

- `potential_V`: `E(V)` 측정 potential
- `current_A`: `I(A)` 측정 current
- `elapsed_time_s`: `Elapsed Time(s)`
- `point_number`: 원본 `Point #`
- `source_segment_number`: 원본 `Segment #`
- `segment_number`: `<SegmentN>` 블록 번호
- `e_applied_V`: 가능한 경우 `E Applied(V)`

손상된 행이나 E/I가 숫자가 아닌 행은 해당 행만 제외합니다. 제외 개수와 이유는 UI와 결과 경고에 남습니다. scientific notation을 지원합니다.

실제 `dummy_data` VersaStudio 파일에서는 하나의 `<Segment1>` 블록 안에서 `Segment #`가 0–3으로 바뀌며 네 cycle을 나타냈고, `Definition=` 끝의 숫자 `0`은 실제 데이터 필드가 아닌 schema sentinel이었습니다. Parser는 이 sentinel을 제외하고 in-record `Segment #`를 우선 보존합니다. Sweep 방향 판별에는 noise가 있는 측정 `E(V)`보다 `E Applied(V)`를 우선 사용하지만, peak와 plot의 potential은 계속 측정값 `E(V)`를 사용합니다.

## 농도 입력 원칙

농도는 `.par` 파일에 저장되어 있지 않으므로 업로드 직후 항상 비어 있습니다. `1mM.par`이나 `2.5mM.par` 같은 파일명에서도 농도를 추출하지 않습니다. 사용자가 표에 값을 직접 입력하고 **Apply inputs**를 눌러야 합니다.

M, mM, µM, nM은 내부적으로 calibration의 공통 mM 단위로 변환됩니다. ppm 또는 Custom은 자동 환산되지 않으며 SI 단위와 섞으면 회귀에서 제외하고 경고합니다. 농도가 비어 있거나 음수/유효하지 않은 파일도 회귀에서 제외되지만 결과 CSV에는 남습니다. 0 농도도 입력할 수 있습니다.

## cycle 및 sweep 판별

점 수를 cycle 수로 균등 분할하지 않습니다.

1. parser가 보존한 `<SegmentN>` 경계를 우선 사용합니다.
2. 각 segment에서 연속 `E(V)` 변화의 부호를 계산해 anodic/cathodic/hold를 지정합니다.
3. 반복 switching-potential 점은 주변 sweep과 결합하고 지속 plateau는 hold로 유지합니다.
4. initial direction → opposite direction → initial direction 복귀를 새 cycle 시작으로 판단합니다.
5. `Cycles` 메타데이터는 검증용 hint로 보존하며 점을 강제로 균등 분할하는 데 사용하지 않습니다.

지원 선택은 First cycle, Last cycle, 특정 Cycle N, Last N cycles average, All cycles separately입니다.

## 로컬 실행

Python 3.11 이상이 필요합니다.

```bash
cd cv_web_analyzer
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 `http://localhost:8501`을 엽니다.

## 테스트

```bash
pytest -q
```

테스트는 metadata/segment/E/I/scientific notation/손상 행 파싱, 파일명 농도 비자동입력, hash ID, cycle/sweep, last cycle, peak window와 boundary 경고, fixed-potential 보간, charge 적분, scan-rate 변환, 회귀, replicate, blank, CSV/ZIP을 검증합니다. 테스트 fixture는 VersaStudio의 전역 metadata + 여러 `<SegmentN>` + `Definition=` 구조를 재현합니다.

## Streamlit Community Cloud 배포

1. 이 폴더의 내용을 GitHub repository 루트에 push합니다.
2. [Streamlit Community Cloud](https://share.streamlit.io/)에 로그인하고 **Create app**을 선택합니다.
3. repository, branch, `app.py`를 지정합니다.
4. 필요하면 **Advanced settings → Secrets**에 아래 내용을 추가합니다.

```toml
APP_PASSWORD = "강한-비밀번호"
```

5. Deploy를 누릅니다. 빌드가 끝나면 `https://<app-name>.streamlit.app` URL을 사용자에게 공유합니다.

Community Cloud의 resource limit은 업로드 데이터 크기와 동시 사용자 수에 영향을 줄 수 있습니다. 연구 데이터나 민감한 데이터에는 인증된 사설 배포를 권장합니다.

## Render / Railway

GitHub repository를 연결한 후 Python 3.11+ service로 배포할 수 있습니다.

- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true`
- 선택 환경변수: `APP_PASSWORD`

플랫폼의 persistent disk는 필요하지 않습니다. 원본 `.par` 파일은 database나 application storage에 저장하지 않습니다.

## Docker 배포

```bash
docker build -t cv-web-analyzer .
docker run --rm -p 8501:8501 cv-web-analyzer
```

인증을 켜려면:

```bash
docker run --rm -p 8501:8501 -e APP_PASSWORD='change-this-password' cv-web-analyzer
```

접속 주소는 `http://localhost:8501`입니다.

## 사설 Linux 서버

가상환경과 requirements를 준비한 뒤 다음과 같이 실행합니다.

```bash
export APP_PASSWORD='change-this-password'   # 선택 사항
streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true
```

운영 환경에서는 systemd, Docker restart policy, 또는 process supervisor로 Streamlit을 관리하십시오. 방화벽에서 8501을 직접 공개하기보다는 Nginx reverse proxy와 HTTPS를 사용하십시오.

예시 `/etc/nginx/sites-available/cv-analyzer`:

```nginx
server {
    listen 80;
    server_name cv-analysis.example.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

설정을 활성화하고 검증합니다.

```bash
sudo ln -s /etc/nginx/sites-available/cv-analyzer /etc/nginx/sites-enabled/cv-analyzer
sudo nginx -t
sudo systemctl reload nginx
```

DNS가 서버를 가리킨 뒤 Debian/Ubuntu에서는 일반적으로 다음과 같이 Let's Encrypt 인증서를 적용할 수 있습니다.

```bash
sudo apt update
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d cv-analysis.example.com
sudo certbot renew --dry-run
```

## 인증과 파일 보안

`APP_PASSWORD`가 환경변수 또는 `.streamlit/secrets.toml`에 있을 때만 로그인 화면이 나타납니다. 비밀번호는 코드나 로그에 기록하지 않습니다. 환경변수가 없으면 URL로 바로 접속할 수 있습니다.

> Uploaded files are processed for the active session and are not intentionally stored permanently.

업로드는 메모리에서 처리되며 원본 `.par`을 database, 장기 server disk, 외부 API에 보내지 않습니다. 원시 CV 데이터는 앱 로그에 출력하지 않습니다. 분석 결과는 사용자가 브라우저 download로 가져갑니다.

Streamlit `session_state`는 강한 보안 저장소가 아닙니다. 민감한 데이터 서비스에는 SSO/reverse-proxy 인증, TLS, 접속 통제, 사설 서버, 적절한 session timeout과 운영 로그 정책을 함께 적용하십시오.
