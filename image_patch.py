from __future__ import annotations

import numpy as np
import re
import torch
from dateutil import parser as dateparser
from PIL import Image
from rich.console import Console
from torchvision import transforms
from torchvision.ops import box_iou
from typing import Union, List
from word2number import w2n

from utils import show_single_image, load_json
from vision_processes import forward, config
import requests
from groundingdino.util.inference import annotate

console = Console(highlight=False)


class ImagePatch:
    """A Python class containing a crop of an image centered around a particular object, as well as relevant
    information.
    Attributes
    ----------
    cropped_image : array_like
        An array-like of the cropped image taken from the original image.
    left : int
        An int describing the position of the left border of the crop's bounding box in the original image.
    lower : int
        An int describing the position of the bottom border of the crop's bounding box in the original image.
    right : int
        An int describing the position of the right border of the crop's bounding box in the original image.
    upper : int
        An int describing the position of the top border of the crop's bounding box in the original image.

    Methods
    -------
    find(object_name: str)->List[ImagePatch]
        Returns a list of new ImagePatch objects containing crops of the image centered around any objects found in the
        image matching the object_name.
    exists(object_name: str)->bool
        Returns True if the object specified by object_name is found in the image, and False otherwise.
    verify_property(property: str)->bool
        Returns True if the property is met, and False otherwise.
    best_text_match(option_list: List[str], prefix: str)->str
        Returns the string that best matches the image.
    simple_query(question: str=None)->str
        Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?".
    compute_depth()->float
        Returns the median depth of the image crop.
    crop(left: int, lower: int, right: int, upper: int)->ImagePatch
        Returns a new ImagePatch object containing a crop of the image at the given coordinates.
    """

    def __init__(self, image: Union[Image.Image, torch.Tensor, np.ndarray], left: int = None, lower: int = None,
                 right: int = None, upper: int = None, parent_left=0, parent_lower=0, queues=None,
                 parent_img_patch=None):
        """Initializes an ImagePatch object by cropping the image at the given coordinates and stores the coordinates as
        attributes. If no coordinates are provided, the image is left unmodified, and the coordinates are set to the
        dimensions of the image.

        Parameters
        -------
        image : array_like
            An array-like of the original image.
        left : int
            An int describing the position of the left border of the crop's bounding box in the original image.
        lower : int
            An int describing the position of the bottom border of the crop's bounding box in the original image.
        right : int
            An int describing the position of the right border of the crop's bounding box in the original image.
        upper : int
            An int describing the position of the top border of the crop's bounding box in the original image.

        """

        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image).permute(1, 2, 0)
        elif isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
            image = image / 255

        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.left = 0
            self.lower = 0
            self.right = image.shape[2]  # width
            self.upper = image.shape[1]  # height
        else:
            self.cropped_image = image[:, image.shape[1]-upper:image.shape[1]-lower, left:right]
            self.left = left + parent_left
            self.upper = upper + parent_lower
            self.right = right + parent_left
            self.lower = lower + parent_lower

        self.height = self.cropped_image.shape[1]
        self.width = self.cropped_image.shape[2]

        self.cache = {}
        self.queues = (None, None) if queues is None else queues

        self.parent_img_patch = parent_img_patch

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2

        if self.cropped_image.shape[1] == 0 or self.cropped_image.shape[2] == 0:
            raise Exception("ImagePatch has no area")

        self.possible_options = load_json('./useful_lists/possible_options.json')

    def forward(self, model_name, *args, **kwargs):
        return forward(model_name, *args, queues=self.queues, **kwargs)

    @property
    def original_image(self):
        if self.parent_img_patch is None:
            return self.cropped_image
        else:
            return self.parent_img_patch.original_image

    def find(self, object_name: str, Box_Threshold: int = None, return_labels: bool = False):
        """Returns a list of ImagePatch objects matching object_name contained in the crop if any are found.
        Otherwise, returns an empty list.
        Parameters
        ----------
        object_name : str
            the name of the object to be found

        Returns
        -------
        List[ImagePatch]
            a list of ImagePatch objects matching object_name contained in the crop
        """
        if config.load_models.GD:
            if return_labels:
                all_object_coordinates, boxes, logits, phrases, image_source = self.forward('GD', self.cropped_image,
                                                                                            object_name, Box_Threshold,
                                                                                            return_labels=return_labels)
            else:
                all_object_coordinates = self.forward('GD', self.cropped_image, object_name, Box_Threshold)

        else:
            if object_name in ["object", "objects"]:
                all_object_coordinates = self.forward('maskrcnn', self.cropped_image)[0]
            else:

                if object_name == 'person':
                    object_name = 'people'  # GLIP does better at people than person

                all_object_coordinates = self.forward('glip', self.cropped_image, object_name)
        if len(all_object_coordinates) == 0:
            if config.load_models.GD and return_labels:
                return [], torch.tensor([[0, 0, 0, 0]]), [], [], image_source
            return []

        threshold = config.ratio_box_area_to_image_area
        if threshold > 0:
            area_im = self.width * self.height
            all_areas = torch.tensor([(coord[2]-coord[0]) * (coord[3]-coord[1]) / area_im
                                      for coord in all_object_coordinates])
            mask = all_areas > threshold
            # if not mask.any():
            #     mask = all_areas == all_areas.max()  # At least return one element
            all_object_coordinates = all_object_coordinates[mask]

        patches = [self.crop(*coordinates) for coordinates in all_object_coordinates]
        if config.load_models.GD and return_labels:
            return patches, boxes, logits, phrases, image_source
        else:
            return patches

    def exists(self, object_name) -> bool:
        """Returns True if the object specified by object_name is found in the image, and False otherwise.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        """
        if object_name.isdigit() or object_name.lower().startswith("number"):
            object_name = object_name.lower().replace("number", "").strip()

            object_name = w2n.word_to_num(object_name)
            answer = self.simple_query("What number is written in the image (in digits)?")
            return w2n.word_to_num(answer) == object_name

        patches = self.find(object_name)

        filtered_patches = []
        for patch in patches:
            patch.print_image()
            if "yes" in patch.simple_query(f"Is this a {object_name}?"):
                filtered_patches.append(patch)
        return len(filtered_patches) > 0

    def get_keyword(self):
        keyword_dict = {}
        keyword = self.forward('ram', self.cropped_image)
        keyword_E = keyword[0].replace('|', ',').replace(' ', '')
        keyword_C = keyword[1].replace('|', ',').replace(' ', '').replace('/', ',')
        outStr = []
        setData = set()
        for word in keyword_E.split(','):
            if word not in setData:
                outStr.append(word)
                setData.add(word)

        return outStr

    def ocr(self):
        ocr = self.forward('easyocr', self.cropped_image)
        return ocr

    def _score(self, category: str, negative_categories=None, model='clip') -> float:
        """
        Returns a binary score for the similarity between the image and the category.
        The negative categories are used to compare to (score is relative to the scores of the negative categories).
        """
        if model == 'clip':
            res = self.forward('clip', self.cropped_image, category, task='score',
                               negative_categories=negative_categories)
        elif model == 'tcl':
            res = self.forward('tcl', self.cropped_image, category, task='score')
        else:  # xvlm
            task = 'binary_score' if negative_categories is not None else 'score'
            res = self.forward('xvlm', self.cropped_image, category, task=task, negative_categories=negative_categories)
            res = res.item()

        return res

    def _detect(self, category: str, thresh, negative_categories=None, model='clip') -> bool:
        return self._score(category, negative_categories, model) > thresh

    def verify_property(self, object_name: str, attribute: str) -> bool:
        """Returns True if the object possesses the property, and False otherwise.
        Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead
        checking whether the object possesses the property.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        attribute : str
            A string describing the property to be checked.
        """
        name = f"{attribute} {object_name}"
        model = config.verify_property.model
        negative_categories = [f"{att} {object_name}" for att in self.possible_options['attributes']]
        if model == 'clip':
            return self._detect(name, negative_categories=negative_categories,
                                thresh=config.verify_property.thresh_clip, model='clip')
        elif model == 'tcl':
            return self._detect(name, thresh=config.verify_property.thresh_tcl, model='tcl')
        else:  # 'xvlm'
            return self._detect(name, negative_categories=negative_categories,
                                thresh=config.verify_property.thresh_xvlm, model='xvlm')

    def best_text_match(self, option_list: list[str] = None, prefix: str = None) -> str:
        """Returns the string that best matches the image.
        Parameters
        -------
        option_list : str
            A list with the names of the different options
        prefix : str
            A string with the prefixes to append to the options
        """
        option_list_to_use = option_list
        if prefix is not None:
            option_list_to_use = [prefix + " " + option for option in option_list]

        model_name = config.best_match_model
        image = self.cropped_image
        text = option_list_to_use
        if model_name in ('clip', 'tcl'):
            selected = self.forward(model_name, image, text, task='classify')
        elif model_name == 'xvlm':
            res = self.forward(model_name, image, text, task='score')
            res = res.argmax().item()
            selected = res
        else:
            raise NotImplementedError

        return option_list[selected]

    def simple_query(self, question: str):
        """Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?". The questions are about basic perception, and are not meant to be used for complex reasoning
        or external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.
        """
        return self.forward('blip', self.cropped_image, question, task='qa')

    def compute_depth(self):
        """Returns the median depth of the image crop
        Parameters
        ----------
        Returns
        -------
        float
            the median depth of the image crop
        """
        original_image = self.original_image
        depth_map = self.forward('depth', original_image)
        depth_map = depth_map[original_image.shape[1]-self.upper:original_image.shape[1]-self.lower,
                              self.left:self.right]
        return depth_map.median()  # Ideally some kind of mode, but median is good enough for now

    def crop(self, left: int, lower: int, right: int, upper: int) -> ImagePatch:
        """Returns a new ImagePatch containing a crop of the original image at the given coordinates.
        Parameters
        ----------
        left : int
            the position of the left border of the crop's bounding box in the original image
        lower : int
            the position of the bottom border of the crop's bounding box in the original image
        right : int
            the position of the right border of the crop's bounding box in the original image
        upper : int
            the position of the top border of the crop's bounding box in the original image

        Returns
        -------
        ImagePatch
            a new ImagePatch containing a crop of the original image at the given coordinates
        """
        # make all inputs ints
        left = int(left)
        lower = int(lower)
        right = int(right)
        upper = int(upper)

        if config.crop_larger_margin:
            left = max(0, left - 10)
            lower = max(0, lower - 10)
            right = min(self.width, right + 10)
            upper = min(self.height, upper + 10)

        return ImagePatch(self.cropped_image, left, lower, right, upper, self.left, self.lower, queues=self.queues,
                          parent_img_patch=self)

    def overlaps_with(self, left, lower, right, upper):
        """Returns True if a crop with the given coordinates overlaps with this one,
        else False.
        Parameters
        ----------
        left : int
            the left border of the crop to be checked
        lower : int
            the lower border of the crop to be checked
        right : int
            the right border of the crop to be checked
        upper : int
            the upper border of the crop to be checked

        Returns
        -------
        bool
            True if a crop with the given coordinates overlaps with this one, else False
        """
        return self.left <= right and self.right >= left and self.lower <= upper and self.upper >= lower

    def llm_query(self, question: str, long_answer: bool = True) -> str:
        return llm_query(question, None, long_answer)

    def print_image(self, size: tuple[int, int] = None):
        show_single_image(self.cropped_image, size)

    def __repr__(self):
        return "ImagePatch({}, {}, {}, {})".format(self.left, self.lower, self.right, self.upper)


def best_image_match(list_patches: list[ImagePatch], content: List[str], return_index: bool = False) -> \
        Union[ImagePatch, None]:
    """Returns the patch most likely to contain the content.
    Parameters
    ----------
    list_patches : List[ImagePatch]
    content : List[str]
        the object of interest
    return_index : bool
        if True, returns the index of the patch most likely to contain the object

    Returns
    -------
    int
        Patch most likely to contain the object
    """
    if len(list_patches) == 0:
        return None

    model = config.best_match_model

    scores = []
    for cont in content:
        if model == 'clip':
            res = list_patches[0].forward(model, [p.cropped_image for p in list_patches], cont, task='compare',
                                          return_scores=True)
        else:
            res = list_patches[0].forward(model, [p.cropped_image for p in list_patches], cont, task='score')
        scores.append(res)
    scores = torch.stack(scores).mean(dim=0)
    scores = scores.argmax().item()  # Argmax over all image patches

    if return_index:
        return scores
    return list_patches[scores]


def distance(patch_a: Union[ImagePatch, float], patch_b: Union[ImagePatch, float]) -> float:
    """
    Returns the distance between the edges of two ImagePatches, or between two floats.
    If the patches overlap, it returns a negative distance corresponding to the negative intersection over union.
    """

    if isinstance(patch_a, ImagePatch) and isinstance(patch_b, ImagePatch):
        a_min = np.array([patch_a.left, patch_a.lower])
        a_max = np.array([patch_a.right, patch_a.upper])
        b_min = np.array([patch_b.left, patch_b.lower])
        b_max = np.array([patch_b.right, patch_b.upper])

        u = np.maximum(0, a_min - b_max)
        v = np.maximum(0, b_min - a_max)

        dist = np.sqrt((u ** 2).sum() + (v ** 2).sum())

        if dist == 0:
            box_a = torch.tensor([patch_a.left, patch_a.lower, patch_a.right, patch_a.upper])[None]
            box_b = torch.tensor([patch_b.left, patch_b.lower, patch_b.right, patch_b.upper])[None]
            dist = - box_iou(box_a, box_b).item()

    else:
        dist = abs(patch_a - patch_b)

    return dist


def bool_to_yesno(bool_answer: bool) -> str:
    """Returns a yes/no answer to a question based on the boolean value of bool_answer.
    Parameters
    ----------
    bool_answer : bool
        a boolean value

    Returns
    -------
    str
        a yes/no answer to a question based on the boolean value of bool_answer
    """
    return "yes" if bool_answer else "no"


def llm_query(query, context=None, long_answer=True, queues=None):
    """Answers a text question using GPT-3. The input question is always a formatted string with a variable in it.

    Parameters
    ----------
    query: str
        the text question to ask. Must not contain any reference to 'the image' or 'the photo', etc.
    """
    if long_answer:
        return forward(model_name='gpt3_general', prompt=query, queues=queues)
    else:
        return forward(model_name='gpt3_qa', prompt=[query, context], queues=queues)


def coerce_to_numeric(string, no_string=False):
    """
    This function takes a string as input and returns a numeric value after removing any non-numeric characters.
    If the input string contains a range (e.g. "10-15"), it returns the first value in the range.
    # TODO: Cases like '25to26' return 2526, which is not correct.
    """
    if any(month in string.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 'july',
                                                 'august', 'september', 'october', 'november', 'december']):
        try:
            return dateparser.parse(string).timestamp().year
        except:  # Parse Error
            pass

    try:
        # If it is a word number (e.g. 'zero')
        numeric = w2n.word_to_num(string)
        return numeric
    except ValueError:
        pass

    # Remove any non-numeric characters except the decimal point and the negative sign
    string_re = re.sub("[^0-9\.\-]", "", string)

    if string_re.startswith('-'):
        string_re = '&' + string_re[1:]

    # Check if the string includes a range
    if "-" in string_re:
        # Split the string into parts based on the dash character
        parts = string_re.split("-")
        return coerce_to_numeric(parts[0].replace('&', '-'))
    else:
        string_re = string_re.replace('&', '-')

    try:
        # Convert the string to a float or int depending on whether it has a decimal point
        if "." in string_re:
            numeric = float(string_re)
        else:
            numeric = int(string_re)
    except:
        if no_string:
            raise ValueError
        # No numeric values. Return input
        return string
    return numeric


def local(patch):
    return patch.right, patch.left, patch.upper, patch.lower


def personwithouthelmet(im, Box_Threshold):
    image_patch = ImagePatch(im)
    box = Box_Threshold
    people_patches, boxes, logits, phrases, image_source = image_patch.find('person', box, return_labels=True)
    helmet_patches = image_patch.find('helmet', box)
    Motorcycles_patches = image_patch.find('motorcycles', box)
    lp = len(people_patches)
    lh = len(helmet_patches)
    lm = len(Motorcycles_patches)
    bx = torch.tensor([])
    p = []
    lg = torch.tensor([])
    for i in range(lp):
        b = []
        r, l, u, lo = local(people_patches[i])
        for j in range(lm):
            if Motorcycles_patches[j].overlaps_with(l, lo, r, u):
                b.append('yes')
                c = []
                for k in range(lh):
                    if helmet_patches[k].overlaps_with(l, lo, r, u):
                        c.append('yes')
                # head_patches = people_patches[i].find('head')
                # if len(head_patches) == 1:
                #     hr, hl, hu, hlo = local(head_patches[0])
                #     for k in range(lh):
                #         if helmet_patches[k].overlaps_with(hl, hlo, hr, hu):
                #             c.append('yes')
                # else:
                #     for k in range(lh):
                #         if helmet_patches[k].overlaps_with(l, lo, r, u):
                #             c.append('yes')

        if 'yes' in b and 'yes' not in c:
            lg = torch.cat((lg, logits[i].unsqueeze(0)))
            bx = torch.cat((bx, boxes[i].unsqueeze(0)), dim=0)
            p.append("rider without helmet")

    if len(bx)==0:
        bx = torch.tensor([[0, 0, 0, 0]])

    annotated_frame = annotate(image_source=image_source, boxes=bx, logits=lg, phrases=p)
    return annotated_frame

def multi_vocabulary_with(image_patch, BoxThreshold0, BoxThreshold1, subject, property):
    subject_patches, boxes, logits, phrases, image_source = image_patch.find(subject, BoxThreshold0, return_labels=True, keep=False)
    propertys = property.split(",")
    propertys_dict = {}
    ctn = True
    change_image = True
    sp = []
    bx = torch.tensor([])
    p = []
    lg = torch.tensor([])
    for i in range(len(propertys)):
        var_name = f"propertys{i}"
        obj = propertys[i]
        propertys_dict[var_name] = image_patch.find(obj, BoxThreshold1, keep=False)
        if len(propertys_dict[var_name])==0:
            change_image = False
            if obj=='':
                return ctn, subject_patches, boxes, logits, phrases, image_source, change_image
            ctn = False
            return ctn, subject_patches, boxes, logits, phrases, image_source, change_image
    lp = len(subject_patches)
    for i in range(lp):
        b = True
        r, l, u, lo = local(subject_patches[i])
        for j in range(len(propertys_dict)):
            if not b:
                break
            var_name = f"propertys{j}"
            patches = propertys_dict[var_name]
            for k in range(len(patches)):
                if patches[k].overlaps_with(l, lo, r, u):
                    break
                if k == (len(patches)-1):
                    b = False
        if b:
            sp.append(subject_patches[i])
            lg = torch.cat((lg, logits[i].unsqueeze(0)))
            bx = torch.cat((bx, boxes[i].unsqueeze(0)), dim=0)
            p.append("object")

    if len(bx)==0:
        change_image = False
        bx = torch.tensor([[0, 0, 0, 0]])
    return ctn, sp, bx, lg, p, image_source, change_image

def multi_vocabulary_without(image_patch, sp, bx, lg, pu, image_source, without, BoxThreshold2):
    withouts = without.split(",")
    change_image = True
    withouts_dict = {}
    bx1 = torch.tensor([])
    p = []
    lg1 = torch.tensor([])
    for i in range(len(withouts)):
        var_name = f"propertys{i}"
        obj = withouts[i]
        withouts_dict[var_name] = image_patch.find(obj, BoxThreshold2, keep=False)
        if len(withouts_dict[var_name])==0:
            annotated_frame = annotate(image_source=image_source, boxes=bx, logits=lg, phrases=pu)
            change_image = False
            return annotated_frame[:, :, ::-1], change_image
    lp = len(sp)
    for i in range(lp):
        b = True
        r, l, u, lo = local(sp[i])
        for j in range(len(withouts_dict)):
            if not b:
                break
            var_name = f"propertys{j}"
            patches = withouts_dict[var_name]
            for k in range(len(patches)):
                if patches[k].overlaps_with(l, lo, r, u):
                    b = False
                    break
        if b:
            lg1 = torch.cat((lg1, lg[i].unsqueeze(0)))
            bx1 = torch.cat((bx1, bx[i].unsqueeze(0)), dim=0)
            p.append("object")

    if len(bx1)==0:
        change_image = False
        bx1 = torch.tensor([[0, 0, 0, 0]])

    annotated_frame = annotate(image_source=image_source, boxes=bx1, logits=lg1, phrases=p)
    return annotated_frame[:, :, ::-1], change_image


def multi_vocabulary(im, BoxThreshold0, BoxThreshold1, BoxThreshold2, subject, property, without):
    image_patch = ImagePatch(im)
    ctn, sp, bx, lg, p, image_source, if_change_image1 = multi_vocabulary_with(image_patch,  BoxThreshold0, BoxThreshold1, subject, property)
    if ctn:
        annotated_frame, if_change_image2 = multi_vocabulary_without(image_patch, sp, bx, lg, p, image_source, without, BoxThreshold2)
        if not if_change_image2 and not if_change_image1:
            change_image = False
        else:
            change_image = True
        return annotated_frame, change_image
    else:
        return image_source, if_change_image1


# def get_keyword(image:list or str):
#     if not isinstance(image, list):
#         image = [image]
#     keyword_dict = {}
#     for i, path in enumerate(image):
#         if path.startswith("http://") or path.startswith("https://"):
#             image = Image.open(requests.get(path, stream=True).raw).convert('RGB')
#             image = transforms.ToTensor()(image)
#         else:
#             image = Image.open(path)
#             image = transforms.ToTensor()(image)
#
#         image_patch = ImagePatch(image)
#         keyword = image_patch.keywords()
#         keyword_E = keyword[0].replace('|', ',').replace(' ', '')
#         keyword_C = keyword[1].replace('|', ',').replace(' ', '').replace('/', ',')
#         outStr = []
#         setData = set()
#         for word in keyword_C.split(','):
#             if word not in setData:
#                 outStr.append(word)
#                 setData.add(word)
#         keyword_dict[path] = outStr
#
#     return keyword_dict



