import random
from .constants import (
    LOGOS_TC_TASKS,
    LOGOS_TS_TASKS,
    LOGOS_TCS_TASKS,
    IN_TC_TASKS,
    IN_TS_TASKS,
    IN_TCS_TASKS,
    ICONS_TC_TASKS,
    ICONS_TS_TASKS,
    ICONS_TCS_TASKS,
    LOGOS,
    IN,
    ICONS,
    SCENES,
    ICL_BOTH_TASKS,
    ICL_SCENE_TASKS,
    ICL_SHAPE_TASKS,
    ICL_NEITHER_TASKS
)


def select_demonstration(
    support_meta,
    n_shot,
    dataset,
    query
):
    """
    Selects demonstration examples for few-shot learning based on the dataset and query.

    Args:
        support_meta (Dict[str, Dict[str, List[str]]]): 
            Metadata containing scenes and shapes with corresponding image paths.
            Format: {scene: {shape: [image1, image2, ...], ...}, ...}
        n_shot (int): 
            Number of demonstration examples to select.
        dataset (str): 
            The name of the dataset, determining the type of shapes to use.
        query (Optional[Dict[str, Any]]): 
            The query containing the image and its attributes (scene and shape).
            Required if dataset is in ICL_BOTH_TASKS, ICL_SCENE_TASKS, etc.

    Returns:
        List[Dict[str, Any]]: 
            A list of selected demonstration examples with image paths, scenes, shapes, and answers.

    Raises:
        ValueError: 
            If the dataset is unknown or if an invalid prediction task is specified.
    """
    # Determine the type of shapes based on the dataset
    if "logos" in dataset:
        shapes = LOGOS
    elif "in" in dataset:
        shapes = IN
    elif "icons" in dataset:
        shapes = ICONS
    else:
        raise ValueError("Unknown dataset")

    # Return empty list if no demonstrations are needed
    if n_shot == 0:
        return []

    # Handle datasets requiring both scene and shape identification
    if dataset in ICL_BOTH_TASKS:
        test_image_path = query["image"]
        test_scene = query["scene"]
        test_shape = query["shape"]

        selected_demonstrations = []
        used_images = set([test_image_path])  # Avoid reusing the test image
        used_shapes = set([test_shape])      # Track used shapes
        used_scenes = set([test_scene])      # Track used scenes

        if n_shot == 1:
            # Select one image with the same scene and shape
            possible_images = [
                img for img in support_meta[test_scene][test_shape]
                if img not in used_images
            ]
            if possible_images:
                chosen_image = random.choice(possible_images)
                new_item = {
                    "image": chosen_image,
                    "scene": test_scene,
                    "shape": test_shape,
                }

                # Determine the answer based on task condition
                parts = dataset.split("_")
                if parts[1] == "tc":
                    new_item["answer"] = test_shape
                elif parts[1] == "ts":
                    new_item["answer"] = test_scene
                elif parts[1] == "tcs":
                    new_item["answer"] = f"{test_shape}, {test_scene}"
                else:
                    raise ValueError("Invalid prediction task.")

                selected_demonstrations.append(new_item)
                used_images.add(chosen_image)
                used_shapes.add(test_shape)
                used_scenes.add(test_scene)
        else:
            # Select one example with the same scene but different shape
            available_shapes = [shape for shape in shapes if shape != test_shape]
            if available_shapes:
                chosen_shape = random.choice(available_shapes)
                if chosen_shape in support_meta[test_scene]:
                    possible_images = [
                        img for img in support_meta[test_scene][chosen_shape]
                        if img not in used_images
                    ]
                    if possible_images:
                        chosen_image = random.choice(possible_images)
                        new_item = {
                            "image": chosen_image,
                            "scene": test_scene,
                            "shape": chosen_shape,
                        }

                        parts = dataset.split("_")
                        if parts[1] == "tc":
                            new_item["answer"] = chosen_shape
                        elif parts[1] == "ts":
                            new_item["answer"] = test_scene
                        elif parts[1] == "tcs":
                            new_item["answer"] = f"{test_shape}, {test_scene}"
                        else:
                            raise ValueError("Invalid prediction task.")

                        selected_demonstrations.append(new_item)
                        used_images.add(chosen_image)
                        used_shapes.add(test_shape)
                        used_scenes.add(test_scene)

            # Select one example with the same shape but different scene
            other_scenes = [
                scene for scene in support_meta
                if scene != test_scene and test_shape in support_meta[scene]
            ]
            if other_scenes:
                chosen_scene = random.choice(other_scenes)
                possible_images = [
                    img for img in support_meta[chosen_scene][test_shape]
                    if img not in used_images
                ]
                if possible_images:
                    chosen_image = random.choice(possible_images)
                    new_item = {
                        "image": chosen_image,
                        "scene": chosen_scene,
                        "shape": test_shape,
                    }

                    parts = dataset.split("_")
                    if parts[1] == "tc":
                        new_item["answer"] = test_shape
                    elif parts[1] == "ts":
                        new_item["answer"] = chosen_scene
                    elif parts[1] == "tcs":
                        new_item["answer"] = f"{test_shape}, {chosen_scene}"
                    else:
                        raise ValueError("Invalid prediction task.")

                    selected_demonstrations.append(new_item)
                    used_images.add(chosen_image)
                    used_shapes.add(test_shape)
                    used_scenes.add(chosen_scene)

            # Select additional images ensuring unique shapes and scenes
            num_additional = n_shot - len(selected_demonstrations)
            available_scenes = [
                scene for scene in support_meta if scene not in used_scenes
            ]
            available_shapes = [shape for shape in shapes if shape not in used_shapes]

            for _ in range(num_additional):
                if available_scenes and available_shapes:
                    random_scene = random.choice(available_scenes)
                    available_scenes.remove(random_scene)

                    random_shape = random.choice(available_shapes)
                    available_shapes.remove(random_shape)
                else:
                    # Fallback to any available scene and shape
                    random_scene = random.choice(list(support_meta.keys()))
                    random_shape = random.choice(
                        list(support_meta[random_scene].keys())
                    )

                possible_images = [
                    img for img in support_meta[random_scene][random_shape]
                    if img not in used_images
                ]
                if possible_images:
                    chosen_image = random.choice(possible_images)
                    new_item = {
                        "image": chosen_image,
                        "scene": random_scene,
                        "shape": random_shape,
                    }

                    parts = dataset.split("_")
                    if parts[1] == "tc":
                        new_item["answer"] = random_shape
                    elif parts[1] == "ts":
                        new_item["answer"] = random_scene
                    elif parts[1] == "tcs":
                        new_item["answer"] = f"{random_shape}, {random_scene}"
                    else:
                        raise ValueError("Invalid prediction task.")

                    selected_demonstrations.append(new_item)
                    used_images.add(chosen_image)
                    used_shapes.add(random_shape)
                    used_scenes.add(random_scene)

            # Shuffle the selected demonstrations
            random.shuffle(selected_demonstrations)
            n_shot_support = selected_demonstrations

    # Handle datasets focused on scene identification
    elif dataset in ICL_SCENE_TASKS:
        test_image_path = query["image"]
        test_scene = query["scene"]
        test_shape = query["shape"]

        selected_demonstrations = []
        used_images = set([test_image_path])  # Avoid reusing the test image
        used_shapes = set([test_shape])      # Track used shapes
        used_scenes = set([test_scene])      # Track used scenes

        # Include one example with the same shape and different scene
        if test_scene in support_meta:
            available_shapes = [
                shape for shape in shapes
                if shape != test_shape and shape in support_meta[test_scene]
            ]
            if available_shapes:
                chosen_shape = random.choice(available_shapes)
                possible_images = [
                    img for img in support_meta[test_scene][chosen_shape]
                    if img not in used_images
                ]
                if possible_images:
                    chosen_image = random.choice(possible_images)
                    new_item = {
                        "image": chosen_image,
                        "scene": test_scene,
                        "shape": chosen_shape
                    }

                    parts = dataset.split("_")
                    if parts[1] == "tc":
                        new_item["answer"] = chosen_shape
                    elif parts[1] == "ts":
                        new_item["answer"] = test_scene
                    elif parts[1] == "tcs":
                        # Note: Original code had a potential bug with 'random_scene'
                        new_item["answer"] = f"{chosen_shape}, {test_scene}"
                    else:
                        raise ValueError("Invalid prediction task.")

                    selected_demonstrations.append(new_item)
                    used_images.add(chosen_image)
                    used_shapes.add(chosen_shape)

        # Select additional distinct shapes with distinct scenes
        if n_shot > 1:
            remaining_shapes = [shape for shape in shapes if shape not in used_shapes]
            num_additional = min(len(remaining_shapes), n_shot - 1)
            chosen_shapes = random.sample(remaining_shapes, num_additional)

            available_scenes = [
                scene for scene in support_meta if scene not in used_scenes
            ]

            for shape in chosen_shapes:
                if not available_scenes:
                    break  # No more unique scenes available

                chosen_scene = random.choice(available_scenes)
                possible_images = [
                    img for img in support_meta[chosen_scene][shape]
                    if img not in used_images
                ]
                if possible_images:
                    chosen_image = random.choice(possible_images)
                    new_item = {
                        "image": chosen_image,
                        "scene": chosen_scene,
                        "shape": shape
                    }

                    parts = dataset.split("_")
                    if parts[1] == "tc":
                        new_item["answer"] = shape
                    elif parts[1] == "ts":
                        new_item["answer"] = chosen_scene
                    elif parts[1] == "tcs":
                        new_item["answer"] = f"{shape}, {chosen_scene}"
                    else:
                        raise ValueError("Invalid prediction task.")

                    selected_demonstrations.append(new_item)
                    used_images.add(chosen_image)
                    used_shapes.add(shape)
                    used_scenes.add(chosen_scene)
                    available_scenes.remove(chosen_scene)

        # Shuffle the selected demonstrations
        random.shuffle(selected_demonstrations)
        n_shot_support = selected_demonstrations

    # Handle datasets that require neither scene nor shape focus
    elif dataset in ICL_NEITHER_TASKS:
        test_image_path = query["image"]
        test_scene = query["scene"]
        test_shape = query["shape"]

        selected_demonstrations = []
        used_images = set([test_image_path])  # Avoid reusing the test image
        used_shapes = set([test_shape])      # Track used shapes
        used_scenes = set([test_scene])      # Track used scenes

        # Select scenes and shapes different from the test scene and shape
        other_scenes = [scene for scene in support_meta if scene != test_scene]
        available_shapes = [shape for shape in shapes if shape not in used_shapes]

        for scene in other_scenes:
            if len(selected_demonstrations) >= n_shot:
                break  # Stop if enough demonstrations are selected

            available_shapes = [shape for shape in shapes if shape not in used_shapes]
            if not available_shapes:
                break  # No more unique shapes available

            chosen_shape = random.choice(available_shapes)
            possible_images = [
                img for img in support_meta[scene].get(chosen_shape, [])
                if img not in used_images
            ]
            if possible_images:
                chosen_image = random.choice(possible_images)
                new_item = {
                    "image": chosen_image,
                    "scene": scene,
                    "shape": chosen_shape
                }

                parts = dataset.split("_")
                if parts[1] == "tc":
                    new_item["answer"] = chosen_shape
                elif parts[1] == "ts":
                    new_item["answer"] = scene
                elif parts[1] == "tcs":
                    new_item["answer"] = f"{chosen_shape}, {scene}"
                else:
                    raise ValueError("Invalid prediction task.")

                selected_demonstrations.append(new_item)
                used_images.add(chosen_image)
                used_shapes.add(chosen_shape)
                used_scenes.add(scene)

            # Remove the chosen shape to ensure uniqueness
            available_shapes.remove(chosen_shape)

        # Shuffle the selected demonstrations
        random.shuffle(selected_demonstrations)
        n_shot_support = selected_demonstrations

    # Handle datasets focused on shape identification
    elif dataset in ICL_SHAPE_TASKS:
        test_image_path = query["image"]
        test_scene = query["scene"]
        test_shape = query["shape"]

        selected_demonstrations = []
        used_images = set([test_image_path])  # Avoid reusing the test image
        used_shapes = set([test_shape])      # Track used shapes
        used_scenes = set([test_scene])      # Track used scenes

        # Include one example of the test shape with a different scene
        available_scenes = [
            scene for scene in support_meta
            if scene != test_scene and test_shape in support_meta[scene]
        ]
        if available_scenes:
            chosen_scene = random.choice(available_scenes)
            possible_images = [
                img for img in support_meta[chosen_scene][test_shape]
                if img not in used_images
            ]
            if possible_images:
                chosen_image = random.choice(possible_images)
                new_item = {
                    "image": chosen_image,
                    "scene": chosen_scene,
                    "shape": test_shape
                }

                parts = dataset.split("_")
                if parts[1] == "tc":
                    new_item["answer"] = test_shape
                elif parts[1] == "ts":
                    new_item["answer"] = chosen_scene
                elif parts[1] == "tcs":
                    new_item["answer"] = f"{test_shape}, {chosen_scene}"
                else:
                    raise ValueError("Invalid prediction task.")

                selected_demonstrations.append(new_item)
                used_images.add(chosen_image)
                used_scenes.add(chosen_scene)

        # Select additional distinct shapes with distinct scenes
        if n_shot > 1:
            remaining_shapes = [shape for shape in shapes if shape not in used_shapes]
            num_additional = min(len(remaining_shapes), n_shot - 1)
            chosen_shapes = random.sample(remaining_shapes, num_additional)

            for obj in chosen_shapes:
                available_scenes = [
                    scene for scene in support_meta
                    if obj in support_meta[scene] and scene not in used_scenes
                ]
                if available_scenes:
                    chosen_scene = random.choice(available_scenes)
                    possible_images = [
                        img for img in support_meta[chosen_scene][obj]
                        if img not in used_images
                    ]
                    if possible_images:
                        chosen_image = random.choice(possible_images)
                        new_item = {
                            "image": chosen_image,
                            "scene": chosen_scene,
                            "shape": obj
                        }

                        parts = dataset.split("_")
                        if parts[1] == "tc":
                            new_item["answer"] = obj
                        elif parts[1] == "ts":
                            new_item["answer"] = chosen_scene
                        elif parts[1] == "tcs":
                            new_item["answer"] = f"{obj}, {chosen_scene}"
                        else:
                            raise ValueError("Invalid prediction task.")

                        selected_demonstrations.append(new_item)
                        used_images.add(chosen_image)
                        used_shapes.add(obj)
                        used_scenes.add(chosen_scene)

        # Shuffle the selected demonstrations
        random.shuffle(selected_demonstrations)
        n_shot_support = selected_demonstrations

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return n_shot_support


def get_task_instruction(dataset, args) -> str:
    """
    Generates the task instruction based on the dataset and provided arguments.

    Args:
        dataset (str): The name of the dataset.
        args (Any): Additional arguments required for task instructions.

    Returns:
        str: The formatted task instruction.

    Raises:
        ValueError: If the dataset does not match any known task configurations.
    """
    # Define task configurations
    task_configurations = [
        {
            "datasets": LOGOS_TC_TASKS,
            "object_type": "logo",
            "task_condition": "TC",
            "response_words": 1,
            "template": (
                "This image contains a company's logo integrated into a background, where elements of the background contribute to forming the logo.\n"
                "background options: [{BG_OPTIONS}]\n"
                "logo options: [{OBJ_OPTIONS}]\n"
                "Identify the logo that is represented in the image by choosing among the provided options. "
                "Provide your response by stating only the single, most accurate option that represents the logo in the image. "
                "You have to respond with a single word."
            )
        },
        {
            "datasets": LOGOS_TS_TASKS,
            "object_type": "logo",
            "task_condition": "TS",
            "response_words": 1,
            "template": (
                "This image contains a company's logo integrated into a background, where elements of the background contribute to forming the logo.\n"
                "background options: [{BG_OPTIONS}]\n"
                "logo options: [{OBJ_OPTIONS}]\n"
                "Identify the background that is represented in the image by choosing among the provided options. "
                "Provide your response by stating only the single, most accurate option that represents the background in the image. "
                "You have to respond with a single word."
            )
        },
        {
            "datasets": LOGOS_TCS_TASKS,
            "object_type": "logo",
            "task_condition": "TCS",
            "response_words": 2,
            "template": (
                "This image contains a company's logo integrated into a background, where elements of the background contribute to forming the image.\n"
                "background options: [{BG_OPTIONS}]\n"
                "logo options: [{OBJ_OPTIONS}]\n"
                "Identify BOTH the background AND the logo that are represented in the image by choosing among the provided options. "
                "Provide your response by stating only the single, most accurate options that represent the background and the logo in the image respectively. "
                "You have to respond with two words, one predicting the background and one predicting the logo."
            )
        },
        {
            "datasets": IN_TC_TASKS,
            "object_type": "object",
            "task_condition": "TC",
            "response_words": 1,
            "template": (
                "This image contains an object integrated into a background, where elements of the background contribute to forming the image.\n"
                "background options: [{BG_OPTIONS}]\n"
                "object options: [{OBJ_OPTIONS}]\n"
                "Identify the object that is represented in the image by choosing among the provided options. "
                "Provide your response by stating only the single, most accurate option that represents the object in the image. "
                "You have to respond with a single word."
            )
        },
        {
            "datasets": IN_TS_TASKS,
            "object_type": "object",
            "task_condition": "TS",
            "response_words": 1,
            "template": (
                "This image contains an object integrated into a background, where elements of the background contribute to forming the image.\n"
                "background options: [{BG_OPTIONS}]\n"
                "object options: [{OBJ_OPTIONS}]\n"
                "Identify the background that is represented in the image by choosing among the provided options. "
                "Provide your response by stating only the single, most accurate option that represents the background in the image. "
                "You have to respond with a single word."
            )
        },
        {
            "datasets": IN_TCS_TASKS,
            "object_type": "object",
            "task_condition": "TCS",
            "response_words": 2,
            "template": (
                "This image contains an object integrated into a background, where elements of the background contribute to forming the image.\n"
                "background options: [{BG_OPTIONS}]\n"
                "object options: [{OBJ_OPTIONS}]\n"
                "Identify BOTH the background AND the object that are represented in the image by choosing among the provided options. "
                "Provide your response by stating only the single, most accurate options that represent the background and the object in the image respectively. "
                "You have to respond with two words, one predicting the background and one predicting the object."
            )
        },
        {
            "datasets": ICONS_TC_TASKS,
            "object_type": "icon",
            "task_condition": "TC",
            "response_words": 1,
            "template": (
                "This image contains an icon integrated into a background, where elements of the background contribute to forming the icon.\n"
                "background options: [{BG_OPTIONS}]\n"
                "icon options: [{OBJ_OPTIONS}]\n"
                "Identify the icon that is represented in the image by choosing among the provided options. "
                "Provide your response by stating only the single, most accurate option that represents the icon in the image. "
                "You have to respond with a single word."
            )
        },
        {
            "datasets": ICONS_TS_TASKS,
            "object_type": "icon",
            "task_condition": "TS",
            "response_words": 1,
            "template": (
                "This image contains an icon integrated into a background, where elements of the background contribute to forming the icon.\n"
                "background options: [{BG_OPTIONS}]\n"
                "icon options: [{OBJ_OPTIONS}]\n"
                "Identify the background that is represented in the image by choosing among the provided options. "
                "Provide your response by stating only the single, most accurate option that represents the background in the image. "
                "You have to respond with a single word."
            )
        },
        {
            "datasets": ICONS_TCS_TASKS,
            "object_type": "icon",
            "task_condition": "TCS",
            "response_words": 2,
            "template": (
                "This image contains an icon integrated into a background, where elements of the background contribute to forming the image.\n"
                "background options: [{BG_OPTIONS}]\n"
                "icon options: [{OBJ_OPTIONS}]\n"
                "Identify BOTH the background AND the icon that are represented in the image by choosing among the provided options. "
                "Provide your response by stating only the single, most accurate options that represent the background and the icon in the image respectively. "
                "You have to respond with two words, one predicting the background and one predicting the icon."
            )
        },
        {
            "datasets": ["logos_conditioning"],
            "object_type": "logo",
            "task_condition": "conditioning",
            "response_words": 1,
            "template": (
                "This image contains a company's logo. Identify the logo that is represented in the image by choosing exclusively among the following options: [{OBJ_OPTIONS}]. "
                "Provide your response by stating only the single, most accurate class name that represents the logo. "
                "You have to respond with a single word."
            )
        },
        {
            "datasets": ["in_conditioning"],
            "object_type": "object",
            "task_condition": "conditioning",
            "response_words": 1,
            "template": (
                "This image contains an object integrated into a background, where elements of the background contribute to forming the image. "
                "Identify the object that is represented in the image by choosing exclusively among the following options: [{OBJ_OPTIONS}]. "
                "Provide your response by stating only the single, most accurate class name that represents the object. "
                "You have to respond with a single word."
            )
        },
        {
            "datasets": ["icons_conditioning"],
            "object_type": "icon",
            "task_condition": "conditioning",
            "response_words": 1,
            "template": (
                "This image contains an icon. Identify the icon that is represented in the image by choosing exclusively among the following options: [{OBJ_OPTIONS}]. "
                "Provide your response by stating only the single, most accurate class name that represents the icon. "
                "You have to respond with a single word."
            )
        }
    ]
    
    # Iterate through task configurations to find a matching dataset
    for config in task_configurations:
        if dataset in config["datasets"]:
            instr = config["template"]
            # Replace {OBJ_OPTIONS} based on object_type
            if config["object_type"] == "logo":
                instr = instr.replace("{OBJ_OPTIONS}", ", ".join(LOGOS))
            elif config["object_type"] == "icon":
                instr = instr.replace("{OBJ_OPTIONS}", ", ".join(ICONS))
            elif config["object_type"] == "object":
                instr = instr.replace("{OBJ_OPTIONS}", ", ".join(IN))
            else:
                raise ValueError(f"Unknown object type: {config['object_type']}")
            
            # Replace {BG_OPTIONS} with SCENES
            instr = instr.replace("{BG_OPTIONS}", ", ".join(SCENES))
            return instr
    
    # If no matching dataset is found, raise an error
    raise ValueError(f"Unknown dataset: {dataset}")
