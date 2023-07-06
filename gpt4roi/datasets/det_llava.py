import copy
import json
import os
import pickle
import random

import torch
from numpy import source
from PIL import Image
from torch.utils.data import Dataset

from gpt4roi.train.train import preprocess, preprocess_multimodal

CLASSES = (
    'aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock', 'alcohol',
    'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'antenna',
    'apple', 'applesauce', 'apricot', 'apron', 'aquarium',
    'arctic_(type_of_shoe)', 'armband', 'armchair', 'armoire', 'armor',
    'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer',
    'avocado', 'award', 'awning', 'ax', 'baboon', 'baby_buggy',
    'basketball_backboard', 'backpack', 'handbag', 'suitcase', 'bagel',
    'bagpipe', 'baguet', 'bait', 'ball', 'ballet_skirt', 'balloon',
    'bamboo', 'banana', 'Band_Aid', 'bandage', 'bandanna', 'banjo',
    'banner', 'barbell', 'barge', 'barrel', 'barrette', 'barrow',
    'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap',
    'baseball_glove', 'basket', 'basketball', 'bass_horn', 'bat_(animal)',
    'bath_mat', 'bath_towel', 'bathrobe', 'bathtub', 'batter_(food)',
    'battery', 'beachball', 'bead', 'bean_curd', 'beanbag', 'beanie',
    'bear', 'bed', 'bedpan', 'bedspread', 'cow', 'beef_(food)', 'beeper',
    'beer_bottle', 'beer_can', 'beetle', 'bell', 'bell_pepper', 'belt',
    'belt_buckle', 'bench', 'beret', 'bib', 'Bible', 'bicycle', 'visor',
    'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath',
    'birdcage', 'birdhouse', 'birthday_cake', 'birthday_card',
    'pirate_flag', 'black_sheep', 'blackberry', 'blackboard', 'blanket',
    'blazer', 'blender', 'blimp', 'blinker', 'blouse', 'blueberry',
    'gameboard', 'boat', 'bob', 'bobbin', 'bobby_pin', 'boiled_egg',
    'bolo_tie', 'deadbolt', 'bolt', 'bonnet', 'book', 'bookcase',
    'booklet', 'bookmark', 'boom_microphone', 'boot', 'bottle',
    'bottle_opener', 'bouquet', 'bow_(weapon)', 'bow_(decorative_ribbons)',
    'bow-tie', 'bowl', 'pipe_bowl', 'bowler_hat', 'bowling_ball', 'box',
    'boxing_glove', 'suspenders', 'bracelet', 'brass_plaque', 'brassiere',
    'bread-bin', 'bread', 'breechcloth', 'bridal_gown', 'briefcase',
    'broccoli', 'broach', 'broom', 'brownie', 'brussels_sprouts',
    'bubble_gum', 'bucket', 'horse_buggy', 'bull', 'bulldog', 'bulldozer',
    'bullet_train', 'bulletin_board', 'bulletproof_vest', 'bullhorn',
    'bun', 'bunk_bed', 'buoy', 'burrito', 'bus_(vehicle)', 'business_card',
    'butter', 'butterfly', 'button', 'cab_(taxi)', 'cabana', 'cabin_car',
    'cabinet', 'locker', 'cake', 'calculator', 'calendar', 'calf',
    'camcorder', 'camel', 'camera', 'camera_lens', 'camper_(vehicle)',
    'can', 'can_opener', 'candle', 'candle_holder', 'candy_bar',
    'candy_cane', 'walking_cane', 'canister', 'canoe', 'cantaloup',
    'canteen', 'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino',
    'car_(automobile)', 'railcar_(part_of_a_train)', 'elevator_car',
    'car_battery', 'identity_card', 'card', 'cardigan', 'cargo_ship',
    'carnation', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 'carton',
    'cash_register', 'casserole', 'cassette', 'cast', 'cat', 'cauliflower',
    'cayenne_(spice)', 'CD_player', 'celery', 'cellular_telephone',
    'chain_mail', 'chair', 'chaise_longue', 'chalice', 'chandelier',
    'chap', 'checkbook', 'checkerboard', 'cherry', 'chessboard',
    'chicken_(animal)', 'chickpea', 'chili_(vegetable)', 'chime',
    'chinaware', 'crisp_(potato_chip)', 'poker_chip', 'chocolate_bar',
    'chocolate_cake', 'chocolate_milk', 'chocolate_mousse', 'choker',
    'chopping_board', 'chopstick', 'Christmas_tree', 'slide', 'cider',
    'cigar_box', 'cigarette', 'cigarette_case', 'cistern', 'clarinet',
    'clasp', 'cleansing_agent', 'cleat_(for_securing_rope)', 'clementine',
    'clip', 'clipboard', 'clippers_(for_plants)', 'cloak', 'clock',
    'clock_tower', 'clothes_hamper', 'clothespin', 'clutch_bag', 'coaster',
    'coat', 'coat_hanger', 'coatrack', 'cock', 'cockroach',
    'cocoa_(beverage)', 'coconut', 'coffee_maker', 'coffee_table',
    'coffeepot', 'coil', 'coin', 'colander', 'coleslaw',
    'coloring_material', 'combination_lock', 'pacifier', 'comic_book',
    'compass', 'computer_keyboard', 'condiment', 'cone', 'control',
    'convertible_(automobile)', 'sofa_bed', 'cooker', 'cookie',
    'cooking_utensil', 'cooler_(for_food)', 'cork_(bottle_plug)',
    'corkboard', 'corkscrew', 'edible_corn', 'cornbread', 'cornet',
    'cornice', 'cornmeal', 'corset', 'costume', 'cougar', 'coverall',
    'cowbell', 'cowboy_hat', 'crab_(animal)', 'crabmeat', 'cracker',
    'crape', 'crate', 'crayon', 'cream_pitcher', 'crescent_roll', 'crib',
    'crock_pot', 'crossbar', 'crouton', 'crow', 'crowbar', 'crown',
    'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch',
    'cub_(animal)', 'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup',
    'cupboard', 'cupcake', 'hair_curler', 'curling_iron', 'curtain',
    'cushion', 'cylinder', 'cymbal', 'dagger', 'dalmatian', 'dartboard',
    'date_(fruit)', 'deck_chair', 'deer', 'dental_floss', 'desk',
    'detergent', 'diaper', 'diary', 'die', 'dinghy', 'dining_table', 'tux',
    'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher',
    'dishwasher_detergent', 'dispenser', 'diving_board', 'Dixie_cup',
    'dog', 'dog_collar', 'doll', 'dollar', 'dollhouse', 'dolphin',
    'domestic_ass', 'doorknob', 'doormat', 'doughnut', 'dove', 'dragonfly',
    'drawer', 'underdrawers', 'dress', 'dress_hat', 'dress_suit',
    'dresser', 'drill', 'drone', 'dropper', 'drum_(musical_instrument)',
    'drumstick', 'duck', 'duckling', 'duct_tape', 'duffel_bag', 'dumbbell',
    'dumpster', 'dustpan', 'eagle', 'earphone', 'earplug', 'earring',
    'easel', 'eclair', 'eel', 'egg', 'egg_roll', 'egg_yolk', 'eggbeater',
    'eggplant', 'electric_chair', 'refrigerator', 'elephant', 'elk',
    'envelope', 'eraser', 'escargot', 'eyepatch', 'falcon', 'fan',
    'faucet', 'fedora', 'ferret', 'Ferris_wheel', 'ferry', 'fig_(fruit)',
    'fighter_jet', 'figurine', 'file_cabinet', 'file_(tool)', 'fire_alarm',
    'fire_engine', 'fire_extinguisher', 'fire_hose', 'fireplace',
    'fireplug', 'first-aid_kit', 'fish', 'fish_(food)', 'fishbowl',
    'fishing_rod', 'flag', 'flagpole', 'flamingo', 'flannel', 'flap',
    'flash', 'flashlight', 'fleece', 'flip-flop_(sandal)',
    'flipper_(footwear)', 'flower_arrangement', 'flute_glass', 'foal',
    'folding_chair', 'food_processor', 'football_(American)',
    'football_helmet', 'footstool', 'fork', 'forklift', 'freight_car',
    'French_toast', 'freshener', 'frisbee', 'frog', 'fruit_juice',
    'frying_pan', 'fudge', 'funnel', 'futon', 'gag', 'garbage',
    'garbage_truck', 'garden_hose', 'gargle', 'gargoyle', 'garlic',
    'gasmask', 'gazelle', 'gelatin', 'gemstone', 'generator',
    'giant_panda', 'gift_wrap', 'ginger', 'giraffe', 'cincture',
    'glass_(drink_container)', 'globe', 'glove', 'goat', 'goggles',
    'goldfish', 'golf_club', 'golfcart', 'gondola_(boat)', 'goose',
    'gorilla', 'gourd', 'grape', 'grater', 'gravestone', 'gravy_boat',
    'green_bean', 'green_onion', 'griddle', 'grill', 'grits', 'grizzly',
    'grocery_bag', 'guitar', 'gull', 'gun', 'hairbrush', 'hairnet',
    'hairpin', 'halter_top', 'ham', 'hamburger', 'hammer', 'hammock',
    'hamper', 'hamster', 'hair_dryer', 'hand_glass', 'hand_towel',
    'handcart', 'handcuff', 'handkerchief', 'handle', 'handsaw',
    'hardback_book', 'harmonium', 'hat', 'hatbox', 'veil', 'headband',
    'headboard', 'headlight', 'headscarf', 'headset',
    'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'helmet',
    'heron', 'highchair', 'hinge', 'hippopotamus', 'hockey_stick', 'hog',
    'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'hookah',
    'hornet', 'horse', 'hose', 'hot-air_balloon', 'hotplate', 'hot_sauce',
    'hourglass', 'houseboat', 'hummingbird', 'hummus', 'polar_bear',
    'icecream', 'popsicle', 'ice_maker', 'ice_pack', 'ice_skate',
    'igniter', 'inhaler', 'iPod', 'iron_(for_clothing)', 'ironing_board',
    'jacket', 'jam', 'jar', 'jean', 'jeep', 'jelly_bean', 'jersey',
    'jet_plane', 'jewel', 'jewelry', 'joystick', 'jumpsuit', 'kayak',
    'keg', 'kennel', 'kettle', 'key', 'keycard', 'kilt', 'kimono',
    'kitchen_sink', 'kitchen_table', 'kite', 'kitten', 'kiwi_fruit',
    'knee_pad', 'knife', 'knitting_needle', 'knob', 'knocker_(on_a_door)',
    'koala', 'lab_coat', 'ladder', 'ladle', 'ladybug', 'lamb_(animal)',
    'lamb-chop', 'lamp', 'lamppost', 'lampshade', 'lantern', 'lanyard',
    'laptop_computer', 'lasagna', 'latch', 'lawn_mower', 'leather',
    'legging_(clothing)', 'Lego', 'legume', 'lemon', 'lemonade', 'lettuce',
    'license_plate', 'life_buoy', 'life_jacket', 'lightbulb',
    'lightning_rod', 'lime', 'limousine', 'lion', 'lip_balm', 'liquor',
    'lizard', 'log', 'lollipop', 'speaker_(stereo_equipment)', 'loveseat',
    'machine_gun', 'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)',
    'mallard', 'mallet', 'mammoth', 'manatee', 'mandarin_orange', 'manger',
    'manhole', 'map', 'marker', 'martini', 'mascot', 'mashed_potato',
    'masher', 'mask', 'mast', 'mat_(gym_equipment)', 'matchbox',
    'mattress', 'measuring_cup', 'measuring_stick', 'meatball', 'medicine',
    'melon', 'microphone', 'microscope', 'microwave_oven', 'milestone',
    'milk', 'milk_can', 'milkshake', 'minivan', 'mint_candy', 'mirror',
    'mitten', 'mixer_(kitchen_tool)', 'money',
    'monitor_(computer_equipment) computer_monitor', 'monkey', 'motor',
    'motor_scooter', 'motor_vehicle', 'motorcycle', 'mound_(baseball)',
    'mouse_(computer_equipment)', 'mousepad', 'muffin', 'mug', 'mushroom',
    'music_stool', 'musical_instrument', 'nailfile', 'napkin',
    'neckerchief', 'necklace', 'necktie', 'needle', 'nest', 'newspaper',
    'newsstand', 'nightshirt', 'nosebag_(for_animals)',
    'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 'nutcracker',
    'oar', 'octopus_(food)', 'octopus_(animal)', 'oil_lamp', 'olive_oil',
    'omelet', 'onion', 'orange_(fruit)', 'orange_juice', 'ostrich',
    'ottoman', 'oven', 'overalls_(clothing)', 'owl', 'packet', 'inkpad',
    'pad', 'paddle', 'padlock', 'paintbrush', 'painting', 'pajamas',
    'palette', 'pan_(for_cooking)', 'pan_(metal_container)', 'pancake',
    'pantyhose', 'papaya', 'paper_plate', 'paper_towel', 'paperback_book',
    'paperweight', 'parachute', 'parakeet', 'parasail_(sports)', 'parasol',
    'parchment', 'parka', 'parking_meter', 'parrot',
    'passenger_car_(part_of_a_train)', 'passenger_ship', 'passport',
    'pastry', 'patty_(food)', 'pea_(food)', 'peach', 'peanut_butter',
    'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'wooden_leg',
    'pegboard', 'pelican', 'pen', 'pencil', 'pencil_box',
    'pencil_sharpener', 'pendulum', 'penguin', 'pennant', 'penny_(coin)',
    'pepper', 'pepper_mill', 'perfume', 'persimmon', 'person', 'pet',
    'pew_(church_bench)', 'phonebook', 'phonograph_record', 'piano',
    'pickle', 'pickup_truck', 'pie', 'pigeon', 'piggy_bank', 'pillow',
    'pin_(non_jewelry)', 'pineapple', 'pinecone', 'ping-pong_ball',
    'pinwheel', 'tobacco_pipe', 'pipe', 'pistol', 'pita_(bread)',
    'pitcher_(vessel_for_liquid)', 'pitchfork', 'pizza', 'place_mat',
    'plate', 'platter', 'playpen', 'pliers', 'plow_(farm_equipment)',
    'plume', 'pocket_watch', 'pocketknife', 'poker_(fire_stirring_tool)',
    'pole', 'polo_shirt', 'poncho', 'pony', 'pool_table', 'pop_(soda)',
    'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot', 'potato',
    'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn', 'pretzel',
    'printer', 'projectile_(weapon)', 'projector', 'propeller', 'prune',
    'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'pumpkin', 'puncher',
    'puppet', 'puppy', 'quesadilla', 'quiche', 'quilt', 'rabbit',
    'race_car', 'racket', 'radar', 'radiator', 'radio_receiver', 'radish',
    'raft', 'rag_doll', 'raincoat', 'ram_(animal)', 'raspberry', 'rat',
    'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt',
    'recliner', 'record_player', 'reflector', 'remote_control',
    'rhinoceros', 'rib_(food)', 'rifle', 'ring', 'river_boat', 'road_map',
    'robe', 'rocking_chair', 'rodent', 'roller_skate', 'Rollerblade',
    'rolling_pin', 'root_beer', 'router_(computer_equipment)',
    'rubber_band', 'runner_(carpet)', 'plastic_bag',
    'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'safety_pin',
    'sail', 'salad', 'salad_plate', 'salami', 'salmon_(fish)',
    'salmon_(food)', 'salsa', 'saltshaker', 'sandal_(type_of_shoe)',
    'sandwich', 'satchel', 'saucepan', 'saucer', 'sausage', 'sawhorse',
    'saxophone', 'scale_(measuring_instrument)', 'scarecrow', 'scarf',
    'school_bus', 'scissors', 'scoreboard', 'scraper', 'screwdriver',
    'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seaplane',
    'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark',
    'sharpener', 'Sharpie', 'shaver_(electric)''shaving_cream', 'shawl',
    'shears', 'sheep', 'shepherd_dog', 'sherbert', 'shield', 'shirt',
    'shoe', 'shopping_bag', 'shopping_cart', 'short_pants', 'shot_glass',
    'shoulder_bag', 'shovel', 'shower_head', 'shower_cap',
    'shower_curtain', 'shredder_(for_paper)', 'signboard', 'silo', 'sink',
    'skateboard', 'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole',
    'skirt', 'skullcap', 'sled', 'sleeping_bag', 'sling_(bandage)',
    'slipper_(footwear)', 'smoothie', 'snake', 'snowboard', 'snowman',
    'snowmobile', 'soap', 'soccer_ball', 'sock', 'sofa', 'softball',
    'solar_array', 'sombrero', 'soup', 'soup_bowl', 'soupspoon',
    'sour_cream', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)',
    'spatula', 'spear', 'spectacles', 'spice_rack', 'spider', 'crawfish',
    'sponge', 'spoon', 'sportswear', 'spotlight', 'squid_(food)',
    'squirrel', 'stagecoach', 'stapler_(stapling_machine)', 'starfish',
    'statue_(sculpture)', 'steak_(food)', 'steak_knife', 'steering_wheel',
    'stepladder', 'step_stool', 'stereo_(sound_system)', 'stew', 'stirrer',
    'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove', 'strainer',
    'strap', 'straw_(for_drinking)', 'strawberry', 'street_sign',
    'streetlight', 'string_cheese', 'stylus', 'subwoofer', 'sugar_bowl',
    'sugarcane_(plant)', 'suit_(clothing)', 'sunflower', 'sunglasses',
    'sunhat', 'surfboard', 'sushi', 'mop', 'sweat_pants', 'sweatband',
    'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit', 'sword',
    'syringe', 'Tabasco_sauce', 'table-tennis_table', 'table',
    'table_lamp', 'tablecloth', 'tachometer', 'taco', 'tag', 'taillight',
    'tambourine', 'army_tank', 'tank_(storage_vessel)',
    'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure',
    'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup',
    'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth',
    'telephone_pole', 'telephoto_lens', 'television_camera',
    'television_set', 'tennis_ball', 'tennis_racket', 'tequila',
    'thermometer', 'thermos_bottle', 'thermostat', 'thimble', 'thread',
    'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinfoil',
    'tinsel', 'tissue_paper', 'toast_(food)', 'toaster', 'toaster_oven',
    'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toolbox', 'toothbrush',
    'toothpaste', 'toothpick', 'cover', 'tortilla', 'tow_truck', 'towel',
    'towel_rack', 'toy', 'tractor_(farm_equipment)', 'traffic_light',
    'dirt_bike', 'trailer_truck', 'train_(railroad_vehicle)', 'trampoline',
    'tray', 'trench_coat', 'triangle_(musical_instrument)', 'tricycle',
    'tripod', 'trousers', 'truck', 'truffle_(chocolate)', 'trunk', 'vat',
    'turban', 'turkey_(food)', 'turnip', 'turtle', 'turtleneck_(clothing)',
    'typewriter', 'umbrella', 'underwear', 'unicycle', 'urinal', 'urn',
    'vacuum_cleaner', 'vase', 'vending_machine', 'vent', 'vest',
    'videotape', 'vinegar', 'violin', 'vodka', 'volleyball', 'vulture',
    'waffle', 'waffle_iron', 'wagon', 'wagon_wheel', 'walking_stick',
    'wall_clock', 'wall_socket', 'wallet', 'walrus', 'wardrobe',
    'washbasin', 'automatic_washer', 'watch', 'water_bottle',
    'water_cooler', 'water_faucet', 'water_heater', 'water_jug',
    'water_gun', 'water_scooter', 'water_ski', 'water_tower',
    'watering_can', 'watermelon', 'weathervane', 'webcam', 'wedding_cake',
    'wedding_ring', 'wet_suit', 'wheel', 'wheelchair', 'whipped_cream',
    'whistle', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)',
    'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket',
    'wineglass', 'blinder_(for_horses)', 'wok', 'wolf', 'wooden_spoon',
    'wreath', 'wrench', 'wristband', 'wristlet', 'yacht', 'yogurt',
    'yoke_(animal_equipment)', 'zebra', 'zucchini', 'background')


Hallucination_questions= ['Is there any <class> in this picture?',
'Can you see if there is <class> in this photo?',
'Does this photo contain any <class>?',
'Is <class> present in this image?',
'Are there any signs of <class> in this picture?',
'Can you identify if there is <class> in this photograph?',
'Is there any representation of <class> in this image?',
'Can you tell me if <class> is visible in this photo?',
'Does this picture feature <class>?',
'Are there any indications of <class> in this photograph?',
'Is <class> included in this image?',
'Can you see if <class> is present in this picture?',
'Is there any portrayal of <class> in this photo?',
'Does this image contain any elements of <class>?',
'Can you identify if there is any <class> in this picture?',
'Is <class> captured in this photograph?',
'Are there any traces of <class> in this image?',
'Can you tell me if there are any hints of <class> in this photo?',
'Does this picture show any signs of <class>?',
'Is there any depiction of <class> in this image?']

YES=['Yes, there is definitely <class> in the picture.',
'Absolutely, you can see <class> in the image.',
'Certainly, the photo contains <class>.',
'Yes, <class> is present in this picture.',
'Definitely, there are clear signs of <class> in the photograph.',
'Yes, you can clearly identify <class> in this image.',
'Certainly, there is a representation of <class> in this photo.',
'Yes, <class> is visible in this picture.',
'Without a doubt, this picture features <class>.',
'Yes, there are clear indications of <class> in this photograph.']

NO=[
    'No, there is no <class> in the picture.',
"I'm sorry, but <class> is not present in the image.",
'Unfortunately, the photo does not contain <class>.',
'No, <class> is not visible in this picture.',
"I'm afraid there are no signs of <class> in the photograph.",
'No, <class> cannot be identified in this image.',
"I'm sorry, but there is no representation of <class> in this photo.",
'Unfortunately, <class> is not featured in this picture.',
'No, there are no clear indications of <class> in this photograph.',
"I'm sorry, but there is no depiction of <class> in this image."
]
import mmcv


class DetLLava(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 data_path,
                 ann_path,
                 tokenizer,
                 multimodal_cfg,
                 det_pkl_path=None,
                 score_threshold=0.3,
                 max_det=100,
                 max_len_token=10000
                 ):
        super(DetLLava, self).__init__()

        list_data_dict = json.load(open(ann_path, 'r'))
        det_results = pickle.load(open(det_pkl_path,'rb'))
        self.data_path = data_path
        self.det_pkl_path = det_pkl_path
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.multimodal_cfg = multimodal_cfg
        self.score_threshold = score_threshold
        self.max_det =max_det
        self.max_len_token = max_len_token
        self.det_results_dict = {os.path.join(self.data_path ,item['filename']):item for item in det_results}
        #begin_str = "The <image> describes the entire picture, while <spi_descript> describes specific regions within the image.\n"
        begin_str1 = """The <image> provides an overview of the picture. Here is also some regional information about the image, such as <spi_descript>.\n"""
        begin_str2 = """The <image> provides an overview of the picture. \n"""
        self.begin_str_with_bbox = begin_str1
        self.begin_str_no_bbox = begin_str2
        self.max_len_token = max_len_token
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
       # i = 1
       # print(f"{i}th item")
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        image_file = self.list_data_dict[i]['image']
        image_folder = self.data_path
        processor = self.multimodal_cfg['image_processor']
        image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')

                # TODO ablation this
        image_file = os.path.join(image_folder, image_file)
        pred_bboxes = self.det_results_dict[image_file]['pred_bboxes']
        pred_labels = self.det_results_dict[image_file]['labels']
        ori_bboxes = pred_bboxes
        ori_labels = pred_labels
        pred_labels = pred_labels[pred_bboxes[:, 4] > self.score_threshold]
        pred_bboxes = pred_bboxes[pred_bboxes[:, 4] > self.score_threshold][:,:4]
        w, h = pred_bboxes[:, 2] - pred_bboxes[:, 0], pred_bboxes[:, 3] - pred_bboxes[:, 1]
        filter_small = (w > 0.02) & (h > 0.02)
        pred_bboxes = pred_bboxes[filter_small]
        pred_labels = pred_labels[filter_small]
        pred_bboxes = pred_bboxes[:self.max_det]
        pred_labels = pred_labels[:self.max_det]

        if len(pred_bboxes) == 0:
            pred_bboxes = ori_bboxes[:10][:,:4]
            pred_labels = ori_labels[:10]

        # ori_img = mmcv.imread(image_file, backend='cv2')
        # h,w,c = ori_img.shape
        # vis_bboxes = copy.deepcopy(pred_bboxes)
        # vis_bboxes[:,0::2] = vis_bboxes[:,0::2]*h
        # vis_bboxes[:,1::2] = vis_bboxes[:,1::2]*w
        # mmcv.imshow_det_bboxes(ori_img,
        #                        bboxes=vis_bboxes,
        #                        labels=pred_labels,
        #                        class_names=CLASSES,
        #                        show=False,
        #                        out_file=f"./vis_det/{image_file.split('/')[-1]}"
        #                        )


        image = processor.preprocess(image,
                                         do_center_crop=False,
                                         return_tensors='pt')['pixel_values'][0]

        image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                                    size=(224, 224),
                                                    mode='bilinear',
                                                    align_corners=False).squeeze(0)

        cur_token_len = (image.shape[1]//14) * (image.shape[2]//14)   # FIXME: 14 is hardcoded patch size
        copy_source = copy.deepcopy(sources)
        if random.random() > 0.5:
            no_bbox = True
            copy_source[0]['conversations'][0]['value'] = \
                copy_source[0]['conversations'][0]['value'].replace('<image>', self.begin_str_no_bbox)
        else:
            no_bbox = False
            copy_source[0]['conversations'][0]['value'] = \
                copy_source[0]['conversations'][0]['value'].replace('<image>', self.begin_str_with_bbox)

        spi_string = ''
        for label in pred_labels:
            spi_string = spi_string + f'<bbox> may feature a {CLASSES[label]},'
        assert len(pred_bboxes) == len(pred_labels)
        if not no_bbox:
            copy_source[0]['conversations'][0]['value'] = \
                copy_source[0]['conversations'][0]['value'].replace('<spi_descript>',
                                                            spi_string)
        solve_hallucination = True

        if solve_hallucination:
            see_labels = set(pred_labels.tolist())
            num_labels = len(CLASSES)
            unseen_labels =  set(list(range(0,num_labels))) - see_labels
            select_label = random.randint(0,num_labels-1)
            question = random.choice(Hallucination_questions)

            question = question.replace('<class>', CLASSES[select_label])
            see_flag = False
            if select_label in see_labels:
                see_flag = True
                answer = random.choice(YES).replace('<class>', CLASSES[select_label])
            else:
                answer = random.choice(NO).replace('<class>', CLASSES[select_label])
            sources[0]['conversations'].append(
                {'from': 'human', 'value': question})
            sources[0]['conversations'].append({'from': 'gpt', 'value': answer})
            if see_flag:
                select_label = random.choice(list(unseen_labels))
                question = random.choice(Hallucination_questions)
                question = question.replace('<class>', CLASSES[select_label])
                answer = random.choice(NO).replace('<class>', CLASSES[select_label])
            else:
                select_label = random.choice(list(see_labels))
                question = random.choice(Hallucination_questions)
                question = question.replace('<class>', CLASSES[select_label])
                answer = random.choice(YES).replace('<class>', CLASSES[select_label])


            sources[0]['conversations'].append(
                {'from': 'human', 'value': question})
            sources[0]['conversations'].append({'from': 'gpt', 'value': answer})



        # print(copy_source)
        sources = preprocess_multimodal(
            copy.deepcopy([e['conversations'] for e in copy_source]),
            self.multimodal_cfg, cur_token_len)

        data_dict = preprocess(
            sources,
            self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict['input_ids'][0],
                             labels=data_dict['labels'][0])

        # print(sources)
        data_dict['image'] = image
        if no_bbox:
            data_dict['bboxes'] = torch.zeros(0,4)
        else:
            data_dict['bboxes'] = torch.Tensor(pred_bboxes)


        data_dict['img_metas'] = dict(filename=image_file)


        return data_dict
