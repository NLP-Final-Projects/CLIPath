#!/bin/bash
python -m celery -A web_demo worker -l info -P eventlet -c 1000
