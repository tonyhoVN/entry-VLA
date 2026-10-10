basketball_in_hoop 0
close_box 0
close_fridge 0
close_microwave 0
insert_onto_square_peg 0
open_box 0
press_switch 0
open_drawer 0
open_microwave 0
pick_up_cup 0
push_button 4
put_knife_in_knife_block 0
pour_from_cup_to_cup 0
put_plate_in_colored_dish_rack 0
wipe_desk 0
----
## Requirement
- About is single task and their index of variance (0 index). generate each task 20 trajectory espisodes, each espisode is 1 random seed (Make it in sequence from seed 1 -> seed 20). 
- After that process split phase based on contact. the start manipulation phase will 10 frames before contact event. 
- Limit the CPU (use maximum 4 core) and computation resource for data generation and process to avoid crash.
- Run everything in background.