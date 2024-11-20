from transformers import AutoProcessor, Idefics2Processor

processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics-9b")
print(processor.to_dict())

# dogs_image_url_1 = "https://huggingface.co/datasets/hf-internal-testing/fixtures_nlvr2/raw/main/image1.jpeg"
# dogs_image_url_2 = "https://huggingface.co/datasets/hf-internal-testing/fixtures_nlvr2/raw/main/image2.jpeg"

# prompts = [
#     [
#         "User:",
#         dogs_image_url_1,
#         "Describe this image.\nAssistant: An image of two dogs.\n",
#         "User:",
#         dogs_image_url_2,
#         "Describe this image.\nAssistant:",
#     ]
# ]

# inputs = processor(prompts, return_tensors="pt")

# print("inputs_ids", inputs["input_ids"].shape)
# print("pixel_values", inputs["pixel_values"].shape)

# batch_size = 1
# sequence_length = 128

# num_images = 1
# num_channels = 3
# height = 224
# width = 224

# patch_size = 14
# temporal_patch_size = 2

# input_ids = torch.rand(
#     size=(
#         batch_size,
#         sequence_length,
#     )
# )

# pixel_values = torch.rand(
#     size=(
#         num_images * int(height / patch_size) * int(width / patch_size),
#         num_channels * patch_size * patch_size * temporal_patch_size,
#     )
# )
# image_grid_thw = torch.tensor([[num_images, int(height / patch_size), int(width / patch_size)]])


# print("image_grid_thw", image_grid_thw)
# print("pixel_values", pixel_values.shape)
