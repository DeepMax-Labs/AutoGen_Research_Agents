# filename: text_to_pdf.py
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def save_to_pdf(text, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    for line in text.strip().splitlines():
        height -= 11
        c.drawString(10, height, line.strip())
    c.save()

paper = """YOUR STRING CONTENT"""

save_to_pdf(paper, "research.pdf")