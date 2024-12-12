from fastapi import FastAPI, UploadFile, File, Request, Form, HTTPException, status
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from io import BytesIO
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from fastapi.staticfiles import StaticFiles
from helper import reverse_to_tensor, denormalize_bbox
import os

