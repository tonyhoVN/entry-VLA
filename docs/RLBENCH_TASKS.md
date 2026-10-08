# RLBench tasks in the pinned PerAct2 fork

Source: `ext/RLBench` at markusgrotz/RLBench `8af748c5`. Generated from the task files: 119 single-arm
tasks in `rlbench/tasks/` and 23 bimanual tasks in `rlbench/bimanual_tasks/`. Every task has a scene file in
`rlbench/task_ttms/`. *Variations* is `variation_count()`; symbolic counts such as `len(colors)` were
resolved by importing the module. *Example instruction* is the first string `init_episode` returns, with
`<…>` where the variation fills in a value.

How to generate demos:
- single-arm: `python scripts/rlbench_single_generator.py --task <name> --variation <v> --episodes N --seed S ...`
  (peract2 env);
- bimanual: `python ext/RLBench/tools/dataset_generator_bimanual.py --tasks <name> ...`.

Demos are scripted waypoint plus motion-planning rollouts, not teleoperation.

## Single-arm tasks (robot_setup=panda): 119

| Task | Variations | Example instruction |
|---|---|---|
| `basketball_in_hoop` | 1 | put the ball in the hoop |
| `beat_the_buzz` | 1 | beat the buzz |
| `block_pyramid` | 20 | stack <…> blocks in a pyramid |
| `change_channel` | 2 | turn the channel <…> |
| `change_clock` | 1 | change the clock to show time 12.15 |
| `close_box` | 1 | close box |
| `close_door` | 1 | close the door |
| `close_drawer` | 3 | close <…> drawer |
| `close_fridge` | 1 | close fridge |
| `close_grill` | 1 | close the grill |
| `close_jar` | 20 | close the <…> jar |
| `close_laptop_lid` | 1 | close laptop lid |
| `close_microwave` | 1 | close microwave |
| `empty_container` | 20 | empty the container in the to <…> container |
| `empty_dishwasher` | 1 | empty the dishwasher |
| `get_ice_from_fridge` | 1 | get ice from fridge |
| `hang_frame_on_hanger` | 1 | hang frame on hanger |
| `hit_ball_with_queue` | 1 | hit ball with queue in to the goal |
| `hockey` | 1 | hit the ball into the net |
| `insert_onto_square_peg` | 20 | put the ring on the <…> spoke |
| `insert_usb_in_computer` | 1 | insert usb in computer |
| `lamp_off` | 1 | turn off the light |
| `lamp_on` | 1 | turn on the light |
| `lift_numbered_block` | 3 | pick up the block with the number <…> |
| `light_bulb_in` | 20 | screw in the <…> light bulb |
| `light_bulb_out` | 20 | put the bulb in the <…> holder |
| `meat_off_grill` | 2 | take the <…> off the grill |
| `meat_on_grill` | 2 | put the <…> on the grill |
| `move_hanger` | 1 | move hanger onto the other rackmove the hanger from one rack to the other |
| `open_box` | 1 | open box |
| `open_door` | 1 | open the door |
| `open_drawer` | 3 | open the <…> drawer |
| `open_drawer_test_color_frame` | 3 | open the <…> drawer |
| `open_drawer_test_color_full` | 3 | open the <…> drawer |
| `open_drawer_test_shape_large_handles` | 3 | open the <…> drawer |
| `open_drawer_test_shape_small_body` | 2 | open the <…> drawer |
| `open_drawer_test_shape_square_handles` | 3 | open the <…> drawer |
| `open_drawer_test_texture_frame` | 3 | open the <…> drawer |
| `open_drawer_test_texture_full` | 3 | open the <…> drawer |
| `open_fridge` | 1 | open fridge |
| `open_grill` | 1 | open the grill |
| `open_jar` | 20 | open the <…> jar |
| `open_microwave` | 1 | open microwave |
| `open_oven` | 1 | open the oven |
| `open_washing_machine` | 1 | open washing machine |
| `open_window` | 1 | open left window |
| `open_wine_bottle` | 1 | open wine bottle |
| `phone_on_base` | 1 | put the phone on the base |
| `pick_and_lift` | 20 | pick up the <…> block and lift it up to the target |
| `pick_and_lift_small` | 5 | pick up the <…> and lift it up to the target |
| `pick_up_cup` | 20 | pick up the <…> cup |
| `place_cups` | 3 | place 1 cup on the cup holder |
| `place_hanger_on_rack` | 1 | pick up the hanger and place in on the rackput the hanger on the rack |
| `place_shape_in_shape_sorter` | 5 | put the <…> in the shape sorter |
| `place_wine_at_rack_location` | 3 | stack the wine bottle to the <…> of the rack |
| `play_jenga` | 1 | play jengaTake the protruding block out of the jenga tower without the tower toppling |
| `plug_charger_in_power_supply` | 1 | plug charger in power supply |
| `pour_from_cup_to_cup` | 20 | pour liquid from the <…> cup to the <…> cup |
| `press_switch` | 1 | press switch |
| `push_button` | 18 | push the <…> button |
| `push_buttons` | 50 | rtn0 |
| `put_all_groceries_in_cupboard` | 1 | put all of the groceries in the cupboard |
| `put_books_at_shelf_location` | 4 | setup['lang'] |
| `put_books_on_bookshelf` | 3 | put <…> books on bookshelf |
| `put_bottle_in_fridge` | 1 | put bottle in fridge |
| `put_groceries_in_cupboard` | 9 | put the <…> in the cupboard |
| `put_item_in_drawer` | 3 | put the item in the <…> drawer |
| `put_knife_in_knife_block` | 1 | put the knife in the knife block |
| `put_knife_on_chopping_board` | 1 | put the knife on the chopping board |
| `put_money_in_safe` | 3 | put the money away in the safe on the <…> shelf |
| `put_plate_in_colored_dish_rack` | 20 | put the plate between the <…> pillars of the dish rack |
| `put_rubbish_in_bin` | 1 | put rubbish in bin |
| `put_rubbish_in_color_bin` | 2 | put rubbish in the bin with the same color as the light |
| `put_shoes_in_box` | 1 | put the shoes in the box |
| `put_toilet_roll_on_stand` | 1 | put toilet roll on stand |
| `put_tray_in_oven` | 1 | put tray in oven |
| `put_umbrella_in_umbrella_stand` | 1 | put umbrella in umbrella standpick up the umbrella and drop it in its stand |
| `reach_and_drag` | 20 | use the stick to drag the cube onto the <…> target |
| `reach_target` | 20 | reach the <…> target |
| `remove_cups` | 2 | remove 1 cup from the cup holder and place it on the table |
| `scoop_with_spatula` | 1 | scoop up the cube and lift it with the spatula |
| `screw_nail` | 1 | screw the nail in to the block |
| `set_clock_to_time` | 3 | change the clock to show time 12.<…> |
| `set_the_table` | 1 | set the tableplace the dishes and cutlery on the table in preparation for a meal |
| `setup_checkers` | 3 | (built in code) |
| `setup_chess` | 3 | (built in code) |
| `slide_block_to_color_target` | 4 | slide the block to <…> target |
| `slide_block_to_target` | 1 | slide the block to target |
| `slide_cabinet_open` | 2 | slide <…> cabinet open |
| `slide_cabinet_open_and_place_cups` | 2 | put cup in <…> cabinet |
| `solve_puzzle` | 1 | solve the puzzle |
| `stack_blocks` | 60 | stack <…> <…> blocks |
| `stack_chairs` | 20 | stack the other chairs on top of the <…> chair |
| `stack_cups` | 20 | stack the other cups on top of the <…> cup |
| `stack_wine` | 1 | stack wine bottle |
| `straighten_rope` | 1 | straighten rope |
| `sweep_to_dustpan` | 1 | sweep dirt to dustpan |
| `sweep_to_dustpan_of_size` | 2 | sweep dirt to the <…> dustpan |
| `take_cup_out_from_cabinet` | 2 | take out a cup from the <…> half of the cabinet |
| `take_frame_off_hanger` | 1 | take frame off hanger |
| `take_item_out_of_drawer` | 3 | take item out of the <…> drawer |
| `take_lid_off_saucepan` | 1 | take lid off the saucepan |
| `take_money_out_safe` | 3 | take the money out of the <…> shelf and place it on the  |
| `take_off_weighing_scales` | 3 | remove the <…> pepper from the weighing scales and place it  |
| `take_plate_off_colored_dish_rack` | 3 | take plate off the <…> colored rack |
| `take_shoes_out_of_box` | 1 | take shoes out of box |
| `take_toilet_roll_off_stand` | 1 | take toilet roll off stand |
| `take_tray_out_of_oven` | 1 | take tray out of oven |
| `take_umbrella_out_of_umbrella_stand` | 1 | take umbrella out of umbrella stand |
| `take_usb_out_of_computer` | 1 | take usb out of computer |
| `toilet_seat_down` | 1 | toilet seat down |
| `toilet_seat_up` | 1 | lift toilet seat up |
| `turn_oven_on` | 1 | turn on the oven |
| `turn_tap` | 2 | turn <…> tap |
| `tv_on` | 1 | turn on the TV |
| `unplug_charger` | 1 | unplug charger |
| `water_plants` | 1 | water plant |
| `weighing_scales` | 3 | weigh the <…> pepper |
| `wipe_desk` | 1 | wipe dirt off the desk |

## Bimanual tasks (robot_setup=dual_panda): 23

| Task | Variations | Example instruction |
|---|---|---|
| `bimanual_dual_push_buttons` | 50 | rtn0 |
| `bimanual_handover_item` | 5 | bring me the <…> item |
| `bimanual_handover_item_easy` | 1 | bring me the item |
| `bimanual_lift_ball` | 1 | Lift the ball |
| `bimanual_lift_tray` | 1 | Lift the tray |
| `bimanual_pick_laptop` | 1 | pick up the laptop |
| `bimanual_pick_plate` | 1 | pick up the plate |
| `bimanual_push_box` | 1 | push the box to the red area |
| `bimanual_put_bottle_in_fridge` | 1 | put bottle in fridge |
| `bimanual_put_item_in_drawer` | 3 | put the item in the <…> drawer |
| `bimanual_set_the_table` | 1 | set the tableplace the dishes and cutlery on the table in preparation for a meal |
| `bimanual_straighten_rope` | 1 | straighten rope |
| `bimanual_sweep_to_dustpan` | 1 | sweep dirt to dustpan |
| `bimanual_take_tray_out_of_oven` | 1 | take tray out of oven |
| `coordinated_close_jar` | 20 | close the <…> jar |
| `coordinated_lift_stick` | 1 | lift the stick |
| `coordinated_put_item_in_drawer_right` | 3 | put the item in the <…> drawer |
| `coordinated_take_shoes_out_of_box` | 1 | take shoes out of box |
| `handover_item_medium` | 3 | bring me the <…> item |
| `left_open_drawer` | 3 | open the <…> drawer |
| `right_open_drawer` | 3 | open the <…> drawer |
| `unimanual_push_single_button_left` | 50 | rtn0 |
| `unimanual_push_single_button_right` | 50 | rtn0 |
