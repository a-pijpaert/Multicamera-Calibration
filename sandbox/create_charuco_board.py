import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

def create_charuco_board(output_file, rows, cols, square_length, marker_length):
    # Create ChArUco board
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard((cols, rows), square_length, marker_length, dictionary)

    # Generate board image
    img = board.generateImage((cols*square_length, rows*square_length))

    # Create PDF
    c = canvas.Canvas(output_file, pagesize=(cols*square_length*mm, rows*square_length*mm))
    c.drawInlineImage(img, 0, 0, width=cols*square_length*mm, height=rows*square_length*mm)
    c.showPage()
    c.save()

    print(f"ChArUco board saved as {output_file}")

# Example usage
create_charuco_board("charuco_board.pdf", rows=7, cols=5, square_length=30, marker_length=23)
