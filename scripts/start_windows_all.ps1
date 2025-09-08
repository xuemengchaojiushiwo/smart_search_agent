# PowerShell 一键初始化环境并启动：MySQL 初始化 + Elasticsearch 启动/检查 + Python 服务 + Java 后端
# 用法示例（在仓库根目录下执行）：
#   powershell -ExecutionPolicy Bypass -File .\scripts\start_windows_all.ps1 -MysqlUser root -MysqlPassword yourpass

param(
    [string]$MysqlHost = "127.0.0.1",
    [int]$MysqlPort = 3306,
    [string]$MysqlUser = "root",
    [string]$MysqlPassword = "root",
    [string]$EsUrl = "http://127.0.0.1:9200",
    [int]$PythonPort = 5005,
    [int]$JavaPort = 8080,
    [switch]$SkipEsSetup,
    [switch]$SkipSqlInit
)

$ErrorActionPreference = 'Stop'
function Write-Step($msg) { Write-Host "[STEP] $msg" -ForegroundColor Cyan }
function Write-Ok($msg) { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Test-Command([string]$name) { return [bool](Get-Command $name -ErrorAction SilentlyContinue) }

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

# 1) 基础校验
Write-Step "检查基础工具"
if (-not (Test-Command java)) { throw "未找到 java，请先安装 JDK 17+" }
if (-not (Test-Command mvn)) { throw "未找到 mvn，请先安装 Maven 3.9+" }
if (-not (Test-Command python)) { throw "未找到 python，请先安装 Python 3.10+/3.11" }
if (-not (Test-Command pip)) { throw "未找到 pip，请确保 Python 安装完成" }
if (-not (Test-Command mysql)) { throw "未找到 mysql 客户端，请安装 MySQL 并将其加入 Path" }
Write-Ok "工具已就绪"

# 2) 启动/检查 Elasticsearch
Write-Step "检查 Elasticsearch: $EsUrl"
try {
  $esResp = Invoke-WebRequest -Uri "$EsUrl" -UseBasicParsing -TimeoutSec 3
  Write-Ok "Elasticsearch 已运行"
} catch {
  Write-Warn "Elasticsearch 未响应，尝试执行 start_elasticsearch.bat 启动（如存在）"
  $esBat = Join-Path $RepoRoot "start_elasticsearch.bat"
  if (Test-Path $esBat) {
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c","$esBat" -WindowStyle Minimized | Out-Null
    Write-Warn "已启动 ES，等待端口就绪..."
    Start-Sleep -Seconds 5
    $maxRetry = 24
    for ($i=0; $i -lt $maxRetry; $i++) {
      try { Invoke-WebRequest -Uri "$EsUrl" -UseBasicParsing -TimeoutSec 3 | Out-Null; break } catch { Start-Sleep -Seconds 2 }
    }
    try { Invoke-WebRequest -Uri "$EsUrl" -UseBasicParsing -TimeoutSec 3 | Out-Null; Write-Ok "Elasticsearch 已就绪" } catch { throw "Elasticsearch 启动失败，请手动检查" }
  } else {
    throw "未找到 $esBat，请先手动启动 ES 或调整 -EsUrl"
  }
}

# 3) 初始化数据库 schema
if (-not $SkipSqlInit) {
  Write-Step "初始化 MySQL 数据库: knowledge_base"
  $sqlFile = Resolve-Path (Join-Path $RepoRoot "src/main/resources/db/init_fixed.sql")
  & mysql --host=$MysqlHost --port=$MysqlPort --user=$MysqlUser --password=$MysqlPassword -e "DROP DATABASE IF EXISTS knowledge_base; CREATE DATABASE knowledge_base DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;" | Out-Null
  & mysql --host=$MysqlHost --port=$MysqlPort --user=$MysqlUser --password=$MysqlPassword knowledge_base < $sqlFile
  Write-Ok "数据库初始化完成"
} else { Write-Warn "跳过数据库初始化" }

# 4) Python 服务：安装依赖并启动
Write-Step "启动 Python 服务"
$pyDir = Resolve-Path (Join-Path $RepoRoot "python_service")
Set-Location $pyDir
if (-not (Test-Path ".venv")) {
  python -m venv .venv
}
& .\.venv\Scripts\activate
pip install --upgrade pip | Out-Null
pip install -r requirements.txt | Out-Null

# 4.1 可选：初始化 ES 索引映射
if (-not $SkipEsSetup) {
  Write-Step "重建 ES 索引映射"
  Set-Location $RepoRoot
  .\python_service\.venv\Scripts\python.exe .\scripts\setup_es_knowledge_chunks.py | Out-Null
  Write-Ok "ES 索引映射处理完成"
}

# 4.2 启动 Python app（后台）
Set-Location $pyDir
Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "app_main.py" -WorkingDirectory $pyDir -WindowStyle Minimized | Out-Null
Write-Ok "Python 服务已启动（后台）"

# 5) 启动 Java 后端（后台）
Write-Step "启动 Java 后端"
Set-Location $RepoRoot
Start-Process -FilePath "mvn" -ArgumentList "-DskipTests","spring-boot:run" -WorkingDirectory $RepoRoot -WindowStyle Minimized | Out-Null
Write-Ok "Java 后端已启动（后台）"

# 6) 提示与健康检查
Write-Host ""
Write-Ok "全部步骤完成"
Write-Host "- MySQL:     $MysqlHost:$MysqlPort / knowledge_base" -ForegroundColor Gray
Write-Host "- ES:        $EsUrl" -ForegroundColor Gray
Write-Host "- Python:    http://127.0.0.1:$PythonPort (以代码配置为准)" -ForegroundColor Gray
Write-Host "- Java API:  http://127.0.0.1:$JavaPort" -ForegroundColor Gray
Write-Host "" 
Write-Host "常用验证：" -ForegroundColor Cyan
Write-Host "  GET  /api/knowledge/0/children" -ForegroundColor Gray
Write-Host "  POST /api/search  body:{\"query\":\"安联美元\"}" -ForegroundColor Gray
Write-Host "  GET  /api/chat/sessions?userId=user123" -ForegroundColor Gray
Write-Host ""

Set-Location $RepoRoot

