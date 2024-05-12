import os
import json
from typing import List
from surya.schema import (
    LayoutBox,
    LayoutResult,
    OCRResult,
    TextLine,
    LayoutOCRBox,
    LayoutOCRResult,
)
from surya.postprocessing.textwrap_japanese import fw_fill
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import PyPDF2
import requests


FONT_SIZE = 21
font = ImageFont.truetype(
    "static/fonts/SourceHanSerif-Light.otf",
    size=FONT_SIZE,
)


def find_intersection_lines(
    layout_bbox: LayoutBox, ocr_result: OCRResult
) -> List[TextLine]:
    ret: List[TextLine] = []
    for text in ocr_result.text_lines:
        if layout_bbox.intersection_area(text) / text.area > 0.8:
            ret.append(text)
    return ret


def add_ocr_into_layout(
    layout_predictions: List[LayoutResult], ocr_predictions: List[OCRResult]
) -> List[LayoutOCRResult]:
    ret: List[LayoutOCRResult] = []
    for layout_result, ocr_result in zip(layout_predictions, ocr_predictions):
        layout_ocr_bboxes: List[LayoutOCRBox] = []
        for layout_bbox in layout_result.bboxes:
            if layout_bbox.label != "Text":
                continue
            intersection_ocr_lines = find_intersection_lines(layout_bbox, ocr_result)
            layout_ocr_bbox = LayoutOCRBox(
                label=layout_bbox.label,
                polygon=layout_bbox.polygon,
                confidence=layout_bbox.confidence,
                text_lines=intersection_ocr_lines,
            )
            layout_ocr_bboxes.append(layout_ocr_bbox)
        layout_ocr = LayoutOCRResult(
            bboxes=layout_ocr_bboxes, segmentation_map=None, image_bbox=[]
        )
        ret.append(layout_ocr)
    return ret


def sort_ocr_lines(lines: List[TextLine]) -> List[TextLine]:
    return sorted(lines, key=lambda x: [x.bbox[1], x.bbox[0]])


def translate_text(text: str) -> str:
    url = "http://127.0.0.1:8332"
    data = {
        "content": text,
    }
    resp = requests.post(url, data=json.dumps(data))
    if resp.status_code != 200:
        print("fail to translate text ", resp.content)
        raise Exception("fail to translate text")
    return resp.json()["content"]


def draw_text_on_image(img: np.ndarray, bbox: List[float], text: str) -> np.ndarray:
    processed_text = fw_fill(text, width=int((bbox[2] - bbox[0]) / (FONT_SIZE / 2)))
    new_block = Image.new(
        "RGB",
        (
            int(bbox[2] - bbox[0]),
            int(bbox[3] - bbox[1]),
        ),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(new_block)
    draw.text(
        (0, 0),
        text=processed_text,
        font=font,
        fill=(0, 0, 0),
    )
    new_block = np.array(new_block)
    img[
        int(bbox[1]) : int(bbox[3]),
        int(bbox[0]) : int(bbox[2]),
    ] = new_block
    return img


def save_as_pdf(image: np.ndarray, file_name: str):
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 14))

    # Display the image
    ax.imshow(image, cmap="gray")

    # Remove axes for the image
    plt.axis("off")

    # Save the figure
    plt.savefig(file_name, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def __merge_pdfs(pdf_files: List[str], file_name: str) -> None:
    """Merge translated PDF files into one file.

    Merged file will be stored in the temp directory
    as "translated.pdf".

    Parameters
    ----------
    pdf_files: List[str]
        List of paths to translated PDF files stored in
        the temp directory.
    """
    pdf_merger = PyPDF2.PdfMerger()

    for pdf_file in sorted(pdf_files):
        pdf_merger.append(pdf_file)
    pdf_merger.write(file_name)


# 1. layout 的 text bbox 与 text detection 的 bbox 求交
# 2. 相交的 texts 排序，合并成一个 text
# 3. 翻译 text
# 4. draw translated_text on image
# 5. merge images into a pdf
def draw_orc_on_images(
    result_path: str,
    images: List[Image.Image],
    layout_predictions: List[LayoutResult],
    ocr_predictions: List[OCRResult],
):
    layout_ocr_results = add_ocr_into_layout(layout_predictions, ocr_predictions)
    pdf_files = []
    for idx, (image, layout_ocr) in enumerate(zip(images, layout_ocr_results)):
        img = np.array(image, dtype=np.uint8)
        for layout_ocr_bbox in layout_ocr.bboxes:
            if layout_ocr_bbox.label != "Text":
                continue
            text_lines = sort_ocr_lines(layout_ocr_bbox.text_lines)
            texts = [text_line.text for text_line in text_lines]
            src_text = "".join(texts)
            tgt_text = translate_text(src_text)
            img = draw_text_on_image(img, layout_ocr_bbox.bbox, tgt_text)
        # save image as one pdf
        file_name = os.path.join(result_path, f"{idx:03}.pdf")
        save_as_pdf(img, file_name)
        pdf_files.append(file_name)
    # merge pdfs into one
    result_file = os.path.join(result_path, "translated.pdf")
    __merge_pdfs(pdf_files, result_file)
