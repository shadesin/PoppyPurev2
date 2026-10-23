#!/bin/zsh
cd -- "${0:A:h}" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  print "Run 'uv sync --locked' in PoppyPurev2 first, then open this launcher again."
  read "reply?Press Enter to close. "
  exit 1
fi
.venv/bin/python app/server.py --open "$@"
