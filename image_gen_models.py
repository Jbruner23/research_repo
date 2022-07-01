import vqgan_clip
import subprocess

prompts_manual= {}

caption_style = ["", "in artistic style", "in a painting style", "in a watercolor painting style",
                 ", rendered in the 8k resolution", ", rendered in the Unreal Engine", "in fantasy style",
                 "in oil painting style", "in Disney style"]

prompts_manual['book_name'] = "Alice"
prompts_manual['chapter_1'] = ["a White Rabbit with pink eyes ran close by her",
                               "the Rabbit say to itself, 'Oh dear! Oh dear! I shall be late!'",
                               "the Rabbit actually TOOK A WATCHOUT OF ITS WAISTCOAT-POCKET,"
                               " and looked at it, and then hurried on",
                               "she ran across the field after it, and fortunately was just in time to see it pop down"
                               " a large rabbit-hole under the hedge", "she found herself falling down a very deep well",
                               "she looked at the sides of the well, and noticed that they"
                               " were filled with cupboards and book-shelves; here and there she saw maps and pictures hung upon pegs",
                               "she found herself in a long, low hall, which was lit up by a row of lamps hanging from the roof.",
                               "she came upon a little three-legged table, all made of solid glass; there was nothing on it except a tiny golden key",
                               "she came upon a low curtain she had not noticed before, and behind it was a little door about fifteen inches high: "
                               "she tried the little golden key in the lock, and to her great delight it fitted!",
                               "Alice opened the door and found that it led into a small passage, not much larger than a rat-hole: "
                               "she knelt down and looked along the passage into the loveliest garden you ever saw",
                               "she went back to the table, half hoping she might find another key on it,"
                               " or at any rate a book of rules for shutting people up like telescopes: "
                               "this time she found a little bottle on it, ('which certainly was not here before,' said Alice,)"
                               " and round the neck of the bottle was a paper label, with the words 'DRINK ME' beautifully printed on it in large letters.",
                               "she was now only ten inches high, and her face brightened up at the thought that she was now the right size for"
                               " going through the little door into that lovely garden",
                               "she had forgotten the little golden key, and when she went back to the table for it, she found she could not possibly"
                               " reach it: she could see it quite plainly through the glass, and she tried her best to climb up one of the legs of the table,"
                               "but it was too slippery",
                               "her eye fell on a little glass box that was lying under the table: she opened it, and found in it a very small cake,"
                               " on which the words 'EAT ME' were beautifully marked in currants"]
# prompts_manual['chapter_2'] =

local_out = "/Users/jannabruner/Documents/Janna/MSc_IDC_Computer_Science/research/janna_work/Alice_results"
checkpoint_path = "/Users/jannabruner/Documents/Janna/MSc_IDC_Computer_Science/research/janna_work/vqgan_clip/checkpoints/vqgan_imagenet_f16_16384.ckpt"
config_path = "/Users/jannabruner/Documents/Janna/MSc_IDC_Computer_Science/research/janna_work/vqgan_clip/checkpoints/vqgan_imagenet_f16_16384.yaml"


for i in range(len(prompts_manual['chapter_1'])):
    for j in range(len(caption_style)):
        subprocess.run(["python", "./vqgan_clip/main.py", "-p", str(prompts_manual["chapter_1"][0]), "-ckpt", checkpoint_path,
                "-conf", config_path, "-cd", "cpu", "-i", "2", "-se", "1", "-o", local_out])

vqgan_output = "/home/janna_bruner/results/vqgan"
vqgan_command = "-p  prompt  -cd cpu -i 2 -se 1 -o"