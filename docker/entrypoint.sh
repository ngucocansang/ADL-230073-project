#!/bin/sh
# Waits for Ollama, then starts the web UI (default) or the CLI.
set -e
echo "Waiting for Ollama at ${OLLAMA_URL} ..."
i=0
until curl -fs "${OLLAMA_URL}/api/tags" > /dev/null; do
  i=$((i+1)); [ $i -gt 60 ] && echo "Ollama not reachable" && exit 1
  sleep 2
done
echo "Ollama is up."

case "$1" in
  web)
    exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501 \
         --server.headless=true --browser.gatherUsageStats=false ;;
  cli)
    shift; exec python run_rag.py "$@" ;;
  eval)
    shift; exec python evaluate.py "$@" ;;
  *)
    exec "$@" ;;
esac
