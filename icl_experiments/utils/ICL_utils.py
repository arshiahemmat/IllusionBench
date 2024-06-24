import random
import copy
from .constants import (
    LOGOS,
    LOGOS_TASKS,
    LOGOS_BACKGROUND_TASKS,
    LOGOS_BOTH_TASKS,
    SIN_TASKS,
    SIN_BACKGROUND_TASKS,
    SIN_BOTH_TASKS,
    DOMAINS,
    SIN,
    ICONS,
    ICONS_TASKS,
    ICONS_BACKGROUND_TASKS,
    ICONS_BOTH_TASKS,
    MESSAGES,
    MESSAGES_TASKS,
    MESSAGES_BACKGROUND_TASKS,
    MESSAGES_BOTH_TASKS,
    MESSAGES_DOMAINS,
)


def select_demonstration(support_meta, n_shot, dataset, query=None):
    if "logos" in dataset:
        classes = LOGOS
    elif "sin" in dataset:
        classes = SIN
    elif "icons" in dataset:
        classes = ICONS
    else:
        raise ValueError("Unknown dataset")

    if n_shot == 0:
        return []
    if dataset in [
        "logos_both",
        "logos_background_both",
        "logos_both_both",
        "sin_both",
        "sin_background_both",
        "sin_both_both",
        "icons_both",
        "icons_background_both",
        "icons_both_both",
        "messages_both",
        "messages_background_both",
        "messages_both_both",
    ]:
        test_image_path = query["image"]
        test_background = query["background"]
        if "logos" in dataset:
            test_object = query["logo"]
        elif "sin" in dataset:
            test_object = query["object"]
        elif "icons" in dataset:
            test_object = query["icon"]
        elif "messages" in dataset:
            test_object = query["message"]
        else:
            raise ValueError("Unknown dataset")

        selected_demonstrations = []
        used_images = set([test_image_path])  # Avoid selecting the test image again
        used_objects = set([test_object])  # To track objects used
        used_backgrounds = set([test_background])  # To track backgrounds used

        if n_shot == 1:
            possible_images = [
                img
                for img in support_meta[test_background][test_object]
                if img not in used_images
            ]
            if possible_images:
                chosen_image = random.choice(possible_images)
                new_item = {
                    "image": chosen_image,
                    "background": test_background,
                }
                if "logos" in dataset:
                    new_item["logo"] = test_object
                elif "sin" in dataset:
                    new_item["object"] = test_object
                elif "icons" in dataset:
                    new_item["icon"] = test_object
                elif "messages" in dataset:
                    new_item["message"] = test_object
                else:
                    raise ValueError("Unknown dataset")

                parts = dataset.split("_")
                if len(parts) == 2:
                    new_item["answer"] = test_object
                elif len(parts) == 3:
                    if parts[1] == "background":
                        new_item["answer"] = test_background
                    elif parts[1] == "both":
                        new_item["answer"] = f"{test_object}, {test_background}"
                    else:
                        raise ValueError("Invalid dataset format")

                selected_demonstrations.append(new_item)

                used_images.add(chosen_image)
                used_objects.add(test_object)
                used_backgrounds.add(test_background)
        else:
            # Select one example with the same background and a different object
            available_objects = [cls for cls in classes if cls != test_object]
            if available_objects:
                chosen_object = random.choice(available_objects)
                if chosen_object in support_meta[test_background]:
                    possible_images = [
                        img
                        for img in support_meta[test_background][chosen_object]
                        if img not in used_images
                    ]
                    if possible_images:
                        chosen_image = random.choice(possible_images)
                        new_item = {
                            "image": chosen_image,
                            "background": test_background,
                        }
                        if "logos" in dataset:
                            new_item["logo"] = chosen_object
                        elif "sin" in dataset:
                            new_item["object"] = chosen_object
                        elif "icons" in dataset:
                            new_item["icon"] = chosen_object
                        elif "messages" in dataset:
                            new_item["message"] = chosen_object
                        else:
                            raise ValueError("Unknown dataset")

                        parts = dataset.split("_")
                        if len(parts) == 2:
                            new_item["answer"] = chosen_object
                        elif len(parts) == 3:
                            if parts[1] == "background":
                                new_item["answer"] = test_background
                            elif parts[1] == "both":
                                new_item["answer"] = (
                                    f"{chosen_object}, {test_background}"
                                )
                            else:
                                raise ValueError("Invalid dataset format")

                        selected_demonstrations.append(new_item)

                        used_images.add(chosen_image)
                        used_objects.add(chosen_object)
                        used_backgrounds.add(test_background)

            # Select one example with the same object and a different background
            other_backgrounds = [
                bg
                for bg in support_meta
                if bg != test_background and test_object in support_meta[bg]
            ]
            if other_backgrounds:
                chosen_background = random.choice(other_backgrounds)
                possible_images = [
                    img
                    for img in support_meta[chosen_background][test_object]
                    if img not in used_images
                ]
                if possible_images:
                    chosen_image = random.choice(possible_images)
                    new_item = {
                        "image": chosen_image,
                        "background": chosen_background,
                    }

                    if "logos" in dataset:
                        new_item["logo"] = test_object
                    elif "sin" in dataset:
                        new_item["object"] = test_object
                    elif "icons" in dataset:
                        new_item["icon"] = test_object
                    elif "messages" in dataset:
                        new_item["message"] = test_object

                    parts = dataset.split("_")
                    if len(parts) == 2:
                        new_item["answer"] = test_object
                    elif len(parts) == 3:
                        if parts[1] == "background":
                            new_item["answer"] = chosen_background
                        elif parts[1] == "both":
                            new_item["answer"] = f"{test_object}, {chosen_background}"
                        else:
                            raise ValueError("Invalid dataset format")

                    selected_demonstrations.append(new_item)

                    used_images.add(chosen_image)
                    used_objects.add(test_object)
                    used_backgrounds.add(chosen_background)

            # Select additional images ensuring unique objects and unique backgrounds
            num_additional = n_shot - len(selected_demonstrations)
            available_backgrounds = [
                bg for bg in support_meta if bg not in used_backgrounds
            ]
            available_objects = [obj for obj in classes if obj not in used_objects]

            for _ in range(num_additional):
                if available_backgrounds and available_objects:
                    random_background = random.choice(available_backgrounds)
                    available_backgrounds.remove(random_background)

                    random_class = random.choice(available_objects)
                    available_objects.remove(random_class)
                else:
                    # If no unique backgrounds or objects are left, fallback to any available ones
                    random_background = random.choice(list(support_meta.keys()))
                    random_class = random.choice(
                        list(support_meta[random_background].keys())
                    )

                possible_images = [
                    img
                    for img in support_meta[random_background][random_class]
                    if img not in used_images
                ]
                if possible_images:
                    chosen_image = random.choice(possible_images)
                    new_item = {
                        "image": chosen_image,
                        "background": random_background,
                    }

                    if "logos" in dataset:
                        new_item["logo"] = random_class
                    elif "sin" in dataset:
                        new_item["object"] = random_class
                    elif "icons" in dataset:
                        new_item["icon"] = random_class
                    elif "messages" in dataset:
                        new_item["message"] = random_class

                    parts = dataset.split("_")
                    if len(parts) == 2:
                        new_item["answer"] = random_class
                    elif len(parts) == 3:
                        if parts[1] == "background":
                            new_item["answer"] = random_background
                        elif parts[1] == "both":
                            new_item["answer"] = f"{random_class}, {random_background}"
                        else:
                            raise ValueError("Invalid dataset format")

                    selected_demonstrations.append(new_item)

                    used_images.add(chosen_image)
                    used_objects.add(random_class)
                    used_backgrounds.add(random_background)

        # Shuffle the final set of selected demonstrations
        random.shuffle(selected_demonstrations)

        n_shot_support = selected_demonstrations

    elif dataset in [
        "logos_background",
        "logos_background_background",
        "logos_both_background",
        "sin_background",
        "sin_background_background",
        "sin_both_background",
        "icons_background",
        "icons_background_background",
        "icons_both_background",
        "messages_background",
        "messages_background_background",
        "messages_both_background",
    ]:
        test_image_path = query["image"]
        test_background = query["background"]
        if "logos" in dataset:
            test_object = query["logo"]
        elif "sin" in dataset:
            test_object = query["object"]
        elif "icons" in dataset:
            test_object = query["icon"]
        elif "messages" in dataset:
            test_object = query["message"]
        else:
            raise ValueError("Unknown dataset")

        selected_demonstrations = []
        used_images = set([test_image_path])  # Avoid selecting the test image again
        used_objects = set(
            [test_object]
        )  # Start with the test object to ensure it is not reused
        used_backgrounds = set([test_background])  # Start with the test background

        # Include one example with the same background but a different object
        if test_background in support_meta:
            available_objects = [
                cls
                for cls in classes
                if cls != test_object and cls in support_meta[test_background]
            ]
            if available_objects:
                chosen_object = random.choice(available_objects)
                possible_images = [
                    img
                    for img in support_meta[test_background][chosen_object]
                    if img not in used_images
                ]
                if possible_images:
                    chosen_image = random.choice(possible_images)
                    new_item = {
                        "image": chosen_image,
                        "background": test_background,
                    }

                    if "logos" in dataset:
                        new_item["logo"] = chosen_object
                    elif "sin" in dataset:
                        new_item["object"] = chosen_object
                    elif "icons" in dataset:
                        new_item["icon"] = chosen_object
                    else:
                        raise ValueError("Unknown dataset")

                    parts = dataset.split("_")
                    if len(parts) == 2:
                        new_item["answer"] = chosen_object
                    elif len(parts) == 3:
                        if parts[1] == "background":
                            new_item["answer"] = test_background
                        elif parts[1] == "both":
                            new_item["answer"] = f"{chosen_object}, {test_background}"
                        else:
                            raise ValueError("Invalid dataset format")

                    selected_demonstrations.append(new_item)
                    used_images.add(chosen_image)
                    used_objects.add(chosen_object)

        # Select additional distinct objects with distinct backgrounds
        if n_shot > 1:
            remaining_classes = [cls for cls in classes if cls not in used_objects]
            num_additional = min(
                len(remaining_classes), n_shot - 1
            )  # Ensure we do not need more than available
            chosen_objects = random.sample(remaining_classes, num_additional)

            available_backgrounds = [
                bg for bg in support_meta if bg not in used_backgrounds
            ]

            for obj in chosen_objects:
                for background in available_backgrounds:
                    if obj in support_meta[background]:
                        possible_images = [
                            img
                            for img in support_meta[background][obj]
                            if img not in used_images
                        ]
                        if possible_images:
                            chosen_background = background
                            chosen_image = random.choice(possible_images)

                            new_item = {
                                "image": chosen_image,
                                "background": chosen_background,
                            }

                            if "logos" in dataset:
                                new_item["logo"] = obj
                            elif "sin" in dataset:
                                new_item["object"] = obj
                            elif "icons" in dataset:
                                new_item["icon"] = obj
                            elif "messages" in dataset:
                                new_item["message"] = obj

                            parts = dataset.split("_")
                            if len(parts) == 2:
                                new_item["answer"] = obj
                            elif len(parts) == 3:
                                if parts[1] == "background":
                                    new_item["answer"] = chosen_background
                                elif parts[1] == "both":
                                    new_item["answer"] = f"{obj}, {chosen_background}"
                                else:
                                    raise ValueError("Invalid dataset format")

                            selected_demonstrations.append(new_item)
                            used_images.add(chosen_image)
                            used_objects.add(obj)
                            used_backgrounds.add(chosen_background)
                            available_backgrounds.remove(chosen_background)
                            break

        # Shuffle the final set of selected demonstrations
        random.shuffle(selected_demonstrations)

        n_shot_support = selected_demonstrations

    elif dataset in [
        "logos_neither",
        "logos_background_neither",
        "logos_both_neither",
        "sin_neither",
        "sin_background_neither",
        "sin_both_neither",
        "icons_neither",
        "icons_background_neither",
        "icons_both_neither",
        "messages_neither",
        "messages_background_neither",
        "messages_both_neither",
    ]:
        test_image_path = query["image"]
        test_background = query["background"]
        if "logos" in dataset:
            test_object = query["logo"]
        elif "sin" in dataset:
            test_object = query["object"]
        elif "icons" in dataset:
            test_object = query["icon"]
        elif "messages" in dataset:
            test_object = query["message"]
        else:
            raise ValueError("Unknown dataset")

        selected_demonstrations = []
        used_images = set(
            [test_image_path]
        )  # Start with the test image to avoid selecting it again
        used_objects = set([test_object])  # Avoid using the same object
        used_backgrounds = set([test_background])  # Avoid using the same background

        # Select backgrounds and objects that are different from the test background and object
        other_backgrounds = [bg for bg in support_meta if bg != test_background]
        available_objects = [cls for cls in classes if cls not in used_objects]

        for background in other_backgrounds:
            if len(selected_demonstrations) >= n_shot:
                break  # Stop if we have enough demonstrations

            # Update the available objects each time to ensure they haven't been used in any chosen background
            available_objects = [cls for cls in classes if cls not in used_objects]
            if (
                not available_objects
            ):  # If no more objects are available, break the loop
                break

            chosen_object = random.choice(available_objects)
            possible_images = [
                img
                for img in support_meta[background].get(chosen_object, [])
                if img not in used_images
            ]
            if possible_images:
                chosen_image = random.choice(possible_images)

                new_item = {
                    "image": chosen_image,
                    "background": background,
                }

                if "logos" in dataset:
                    new_item["logo"] = chosen_object
                elif "sin" in dataset:
                    new_item["object"] = chosen_object
                elif "icons" in dataset:
                    new_item["icon"] = chosen_object
                elif "messages" in dataset:
                    new_item["message"] = chosen_object
                else:
                    raise ValueError("Unknown dataset")

                parts = dataset.split("_")
                if len(parts) == 2:
                    new_item["answer"] = chosen_object
                elif len(parts) == 3:
                    if parts[1] == "background":
                        new_item["answer"] = background
                    elif parts[1] == "both":
                        new_item["answer"] = f"{chosen_object}, {background}"
                    else:
                        raise ValueError("Invalid dataset format")

                selected_demonstrations.append(new_item)
                used_images.add(chosen_image)
                used_objects.add(chosen_object)
                used_backgrounds.add(background)

            # Remove the chosen object from available objects after use
            available_objects.remove(chosen_object)

        # Shuffle the final set of selected demonstrations
        random.shuffle(selected_demonstrations)

        n_shot_support = selected_demonstrations

    elif dataset in [
        "logos_logo",
        "logos_background_logo",
        "logos_both_logo",
        "sin_object",
        "sin_background_object",
        "sin_both_object",
        "icons_icon",
        "icons_background_icon",
        "icons_both_icon",
        "messages_message",
        "messages_background_message",
        "messages_both_message",
    ]:
        test_image_path = query["image"]
        test_background = query["background"]
        if "logos" in dataset:
            test_object = query["logo"]
        elif "sin" in dataset:
            test_object = query["object"]
        elif "icons" in dataset:
            test_object = query["icon"]
        elif "messages" in dataset:
            test_object = query["message"]
        else:
            raise ValueError("Unknown dataset")

        selected_demonstrations = []
        used_images = set(
            [test_image_path]
        )  # Start with the test image to avoid reselecting it
        used_objects = set(
            [test_object]
        )  # Start with the test object to ensure it's used once
        used_backgrounds = set(
            [test_background]
        )  # Avoid using the same background for the same object

        # Include one example of the test query object with a different background
        available_backgrounds = [
            bg
            for bg in support_meta
            if bg != test_background and test_object in support_meta[bg]
        ]
        if available_backgrounds:
            chosen_background = random.choice(available_backgrounds)
            possible_images = [
                img
                for img in support_meta[chosen_background][test_object]
                if img not in used_images
            ]
            if possible_images:
                chosen_image = random.choice(possible_images)
                new_item = {
                    "image": chosen_image,
                    "background": chosen_background,
                }

                if "logos" in dataset:
                    new_item["logo"] = test_object
                elif "sin" in dataset:
                    new_item["object"] = test_object
                elif "icons" in dataset:
                    new_item["icon"] = test_object
                elif "messages" in dataset:
                    new_item["message"] = test_object
                else:
                    raise ValueError("Unknown dataset")

                parts = dataset.split("_")
                if len(parts) == 2:
                    new_item["answer"] = test_object
                elif len(parts) == 3:
                    if parts[1] == "background":
                        new_item["answer"] = chosen_background
                    elif parts[1] == "both":
                        new_item["answer"] = f"{test_object}, {chosen_background}"
                    else:
                        raise ValueError("Invalid dataset format")

                selected_demonstrations.append(new_item)
                used_images.add(chosen_image)
                used_backgrounds.add(chosen_background)

        # Now select additional objects that are distinct from the test object
        available_objects = [cls for cls in classes if cls not in used_objects]

        if n_shot > 1:
            num_additional = min(len(available_objects), n_shot - 1)
            chosen_objects = random.sample(available_objects, num_additional)

            for obj in chosen_objects:
                available_backgrounds = [
                    bg
                    for bg in support_meta
                    if obj in support_meta[bg] and bg not in used_backgrounds
                ]
                if available_backgrounds:
                    chosen_background = random.choice(available_backgrounds)
                    possible_images = [
                        img
                        for img in support_meta[chosen_background][obj]
                        if img not in used_images
                    ]
                    if possible_images:
                        chosen_image = random.choice(possible_images)

                        new_item = {
                            "image": chosen_image,
                            "background": chosen_background,
                        }

                        if "logos" in dataset:
                            new_item["logo"] = obj
                        elif "sin" in dataset:
                            new_item["object"] = obj
                        elif "icons" in dataset:
                            new_item["icon"] = obj
                        elif "messages" in dataset:
                            new_item["message"] = obj
                        else:
                            raise ValueError("Unknown dataset")

                        parts = dataset.split("_")
                        if len(parts) == 2:
                            new_item["answer"] = obj
                        elif len(parts) == 3:
                            if parts[1] == "background":
                                new_item["answer"] = chosen_background
                            elif parts[1] == "both":
                                new_item["answer"] = f"{obj}, {chosen_background}"
                            else:
                                raise ValueError("Invalid dataset format")

                        selected_demonstrations.append(new_item)
                        used_images.add(chosen_image)
                        used_objects.add(obj)
                        used_backgrounds.add(chosen_background)

        # Shuffle the final set of selected demonstrations
        random.shuffle(selected_demonstrations)

        n_shot_support = selected_demonstrations

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return n_shot_support


def get_task_instruction(dataset, args):
    description = args.task_description
    if description == "nothing":
        instr = ""
        return instr

    if dataset == "textocr":
        if description == "concise":
            instr = "Answer with the text inside the red box."
        elif description == "detailed":
            instr = "An image will be provided where a red box is drawn around the text of interest. Answer with the text inside the red box. Ensure that the transcription is precise, reflecting the exact characters, including letters, numbers, symbols."
    elif dataset == "operator_induction":
        if description == "concise":
            instr = "Induce the mathematical operator and calculate the result."
        elif description == "detailed":
            instr = "The image contains two digit numbers and a ? representing the mathematical operator. Induce the mathematical operator (addition, multiplication, minus) according to the results of the in-context examples and calculate the result."
    elif dataset == "operator_induction_interleaved":
        if description == "concise":
            instr = "Induce the mathematical operator between the two images and calculate the result."
        elif description == "detailed":
            instr = "There are two input images, each representing a single digit number. Induce the mathematical operator (addition, multiplication, minus) that is applied between the two images according to the results of the in-context examples. Calculate the result for the new query images."
    elif dataset == "open_mi":
        if description == "concise":
            instr = "Answer the question with a single word or phase."
        elif description == "detailed":
            instr = "Induce the concept from the in-context examples. Answer the question with a single word or phase."
    elif dataset == "clevr":
        if description == "concise":
            instr = "Find objects of the given type, induce what operation to use and calculate the result."
        elif description == "detailed":
            instr = "The image contains objects of different shapes, colors, sizes and materials. The question describes the attribute and its value. You need to find all objects within the image that satisfy the condition. You should induce what operation to use according to the results of the in-context examples and then calculate the result."
    elif dataset == "matching_mi":
        if description == "concise":
            instr = "Determine the output for the new pair of images."
        elif description == "detailed":
            instr = 'According to the few-shot examples, induce what operation to do and determine the output for the two new images. Answer with a single token, either "Yes" or "No".'
    if dataset in LOGOS_TASKS:
        instr = "This image contains a company's logo integrated into a background, where elements of the background contribute to forming the logo.\nbackground options: [{BG_OPTIONS}]\nlogo options: [{OBJ_OPTIONS}]\nIdentify the logo that is represented in the image by choosing among the provided options. Provide your response by stating only the single, most accurate option that represents the logo in the image. You have to respond with a single word."
        instr = instr.replace("{OBJ_OPTIONS}", ", ".join(LOGOS))
        instr = instr.replace("{BG_OPTIONS}", ", ".join(DOMAINS))

    elif dataset in LOGOS_BACKGROUND_TASKS:
        instr = "This image contains a company's logo integrated into a background, where elements of the background contribute to forming the logo.\nbackground options: [{BG_OPTIONS}]\nlogo options: [{OBJ_OPTIONS}]\nIdentify the background that is represented in the image by choosing among the provided options. Provide your response by stating only the single, most accurate option that represents the background in the image. You have to respond with a single word."
        instr = instr.replace("{OBJ_OPTIONS}", ", ".join(LOGOS))
        instr = instr.replace("{BG_OPTIONS}", ", ".join(DOMAINS))

    elif dataset in LOGOS_BOTH_TASKS:
        instr = "This image contains a company's logo integrated into a background, where elements of the background contribute to forming the image.\nbackground options: [{BG_OPTIONS}]\nlogo options: [{OBJ_OPTIONS}]\nIdentify BOTH the background AND the logo that are represented in the image by choosing among the provided options. Provide your response by stating only the single, most accurate options that represent the background and the logo in the image respectively. You have to respond with two words, one predicting the background and one predicting the logo."
        instr = instr.replace("{OBJ_OPTIONS}", ", ".join(LOGOS))
        instr = instr.replace("{BG_OPTIONS}", ", ".join(DOMAINS))

    elif dataset in SIN_TASKS:
        instr = "This image contains an object integrated into a background, where elements of the background contribute to forming the image.\nbackground options: [{BG_OPTIONS}]\nobject options: [{OBJ_OPTIONS}]\nIdentify the object that is represented in the image by choosing among the provided options. Provide your response by stating only the single, most accurate option that represents the object in the image. You have to respond with a single word."
        instr = instr.replace("{OBJ_OPTIONS}", ", ".join(SIN))
        instr = instr.replace("{BG_OPTIONS}", ", ".join(DOMAINS))

    elif dataset in SIN_BACKGROUND_TASKS:
        instr = "This image contains an object integrated into a background, where elements of the background contribute to forming the image.\nbackground options: [{BG_OPTIONS}]\nobject options: [{OBJ_OPTIONS}]\nIdentify the background that is represented in the image by choosing among the provided options. Provide your response by stating only the single, most accurate option that represents the background in the image. You have to respond with a single word."
        instr = instr.replace("{OBJ_OPTIONS}", ", ".join(SIN))
        instr = instr.replace("{BG_OPTIONS}", ", ".join(DOMAINS))

    elif dataset in SIN_BOTH_TASKS:
        instr = "This image contains an object integrated into a background, where elements of the background contribute to forming the image.\nbackground options: [{BG_OPTIONS}]\nobject options: [{OBJ_OPTIONS}]\nIdentify BOTH the background AND the object that are represented in the image by choosing among the provided options. Provide your response by stating only the single, most accurate options that represent the background and the object in the image respectively. You have to respond with two words, one predicting the background and one predicting the object."
        instr = instr.replace("{OBJ_OPTIONS}", ", ".join(SIN))
        instr = instr.replace("{BG_OPTIONS}", ", ".join(DOMAINS))

    elif dataset in ICONS_TASKS:
        instr = "This image contains an icon integrated into a background, where elements of the background contribute to forming the icon.\nbackground options: [{BG_OPTIONS}]\nicon options: [{OBJ_OPTIONS}]\nIdentify the icon that is represented in the image by choosing among the provided options. Provide your response by stating only the single, most accurate option that represents the icon in the image. You have to respond with a single word."
        instr = instr.replace("{OBJ_OPTIONS}", ", ".join(ICONS))
        instr = instr.replace("{BG_OPTIONS}", ", ".join(DOMAINS))

    elif dataset in ICONS_BACKGROUND_TASKS:
        instr = "This image contains an icon integrated into a background, where elements of the background contribute to forming the icon.\nbackground options: [{BG_OPTIONS}]\nicon options: [{OBJ_OPTIONS}]\nIdentify the background that is represented in the image by choosing among the provided options. Provide your response by stating only the single, most accurate option that represents the background in the image. You have to respond with a single word."
        instr = instr.replace("{OBJ_OPTIONS}", ", ".join(ICONS))
        instr = instr.replace("{BG_OPTIONS}", ", ".join(DOMAINS))

    elif dataset in ICONS_BOTH_TASKS:
        instr = "This image contains an icon integrated into a background, where elements of the background contribute to forming the image.\nbackground options: [{BG_OPTIONS}]\nicon options: [{OBJ_OPTIONS}]\nIdentify BOTH the background AND the icon that are represented in the image by choosing among the provided options. Provide your response by stating only the single, most accurate options that represent the background and the icon in the image respectively. You have to respond with two words, one predicting the background and one predicting the icon."
        instr = instr.replace("{OBJ_OPTIONS}", ", ".join(ICONS))
        instr = instr.replace("{BG_OPTIONS}", ", ".join(DOMAINS))
    elif dataset in MESSAGES_TASKS:
        instr = "This image contains a message integrated into a background, where elements of the background contribute to forming the message.\nbackground options: [{BG_OPTIONS}]\nmessage options: [{OBJ_OPTIONS}]\nIdentify the message that is represented in the image by choosing among the provided options. Provide your response by stating only the single, most accurate option that represents the message in the image. You have to respond with a single word."
        instr = instr.replace("{OBJ_OPTIONS}", ", ".join(MESSAGES))
        instr = instr.replace("{BG_OPTIONS}", ", ".join(MESSAGES_DOMAINS))
    elif dataset in MESSAGES_BACKGROUND_TASKS:
        instr = "This image contains a message integrated into a background, where elements of the background contribute to forming the message.\nbackground options: [{BG_OPTIONS}]\nmessage options: [{OBJ_OPTIONS}]\nIdentify the background that is represented in the image by choosing among the provided options. Provide your response by stating only the single, most accurate option that represents the background in the image. You have to respond with a single word."
        instr = instr.replace("{OBJ_OPTIONS}", ", ".join(MESSAGES))
        instr = instr.replace("{BG_OPTIONS}", ", ".join(MESSAGES_DOMAINS))
    elif dataset in MESSAGES_BOTH_TASKS:
        instr = "This image contains a message integrated into a background, where elements of the background contribute to forming the image.\nbackground options: [{BG_OPTIONS}]\nmessage options: [{OBJ_OPTIONS}]\nIdentify BOTH the background AND the message that are represented in the image by choosing among the provided options. Provide your response by stating only the single, most accurate options that represent the background and the message in the image respectively. You have to respond with two words, one predicting the background and one predicting the message."
        instr = instr.replace("{OBJ_OPTIONS}", ", ".join(MESSAGES))
        instr = instr.replace("{BG_OPTIONS}", ", ".join(MESSAGES_DOMAINS))
    elif dataset == "logos_conditioning":
        # Choose amoung options instrc.
        instr = "This image contains a company's logo. Identify the logo that is represented in the image by choosing exclusively amoung the following options: {(OPTIONS)}. Provide your response by stating only the single, most accurate class name that represents the logo. You have to respond with a single word."
        instr = instr.replace("(OBJ_OPTIONS)", ", ".join(LOGOS))
        # instr = "This image contains a company's logo. Identify the logo that is represented in the image. Provide your response by stating only the single, most accurate name of the logo. You have to respond with a single word."
    elif dataset == "sin_conditioning":
        instr = "This image contains an object integrated into a background, where elements of the background contribute to forming the image. Identify the object that is represented in the image by choosing exclusively amoung the following options: {(OPTIONS)}. Provide your response by stating only the single, most accurate class name that represents the object. You have to respond with a single word."
        instr = instr.replace("(OBJ_OPTIONS)", ", ".join(SIN))
    elif dataset == "icons_conditioning":
        instr = "This image contains an icon. Identify the icon that is represented in the image by choosing exclusively amoung the following options: {(OPTIONS)}. Provide your response by stating only the single, most accurate class name that represents the icon. You have to respond with a single word."
        instr = instr.replace("(OBJ_OPTIONS)", ", ".join(ICONS))
    elif dataset == "messages_conditioning":
        instr = "This image contains a message. Identify the message that is represented in the image by choosing exclusively amoung the following options: {(OPTIONS)}. Provide your response by stating only the single, most accurate class name that represents the message. You have to respond with a single word."
        instr = instr.replace("(OBJ_OPTIONS)", ", ".join(MESSAGES))
    return instr


def format_answer(answer, dataset, query=None):
    if dataset in ["operator_induction", "clevr", "operator_induction_interleaved"]:
        answer = str(answer)
    return answer
