fn main() {
    let mut policy:[usize;16] = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
    let mut values:[f32;16] = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
    let mut max_value_change = 1.0;
    while max_value_change > 0.01 {
    //for iteration in (0..10){
        (values, max_value_change) = update_values(&policy,values);
        policy = update_policy(policy,&values);
        println!("Max Value Change: {max_value_change}");
        print!("\n\nValues:");
        for position in 0..16 { 
            let value = values[position];
            if position % 4 == 0 {print!("\n")};
            print!("{value:>5.2} ");
        }
        print!("\n\nPolicy:");
        for position in 0..16 {
            let icon = match policy[position] { 
                0 => "🡑 ",
                1 => "🡓 ",
                2 => "🡐 ",
                3 => "🡒 ",
                _ => "Something Terrible Has Happened" //default to informing us that cosmic rays have decided that we are not going to the goal today
            };
            if position % 4 == 0 { print!("\n")}
            print!("{icon}");
        }
    }
}

fn action(state: usize, direction:usize) -> (usize,f32)
 {
    //lookup table for topography of problem, order is [0,1,2,3] = [Up,Down,Left,Right]
    const LINKS: [[usize;4];16] = [[ 0, 4, 0, 1],[ 1, 5, 0, 2],[ 2, 6, 1, 3],[ 3, 7, 2, 3],
                                 [ 0, 8, 4, 5],[ 1, 9, 4, 6],[ 2,10, 5, 7],[ 3,11, 6, 7],
                                 [ 4,12,15, 9],[ 5,13, 8,10],[ 6,14, 9,11],[ 7,15,10,11],
                                 [ 8,12,12,13],[ 9,13,12,14],[10,14,13,15],[11,15,14,15]];
    let new_state:usize = LINKS[state][direction]; //get the new state from the link table
    let reward:f32 = if state == 8 && direction == 2 {-2.0} 
        else if state == 15 && (direction == 1 || direction == 3) {0.0} else {-1.0};
    (new_state,reward) //package the results to return
}

fn update_values(policy:&[usize;16],mut values:[f32;16]) -> ([f32;16],f32) { //, values:[[f32;4];16] 
    let mut max_value_change: f32 = 0.0;
    for location in 0..16 {
        let (new_state,reward) = action(location,policy[location]);
        let new_value: f32 = reward + 0.95 * values[new_state];
        max_value_change = max_value_change.max((values[location]-new_value).abs());
        values[location] = new_value
    };
    (values,max_value_change)
}

fn update_policy(mut policy:[usize;16], values:&[f32;16] ) -> [usize;16] {
    for location in 0..16 {
        let mut best_direction =1;
        let mut best_value:f32 = -999.9;
        for direction in 0..4 {
            let (new_state, _reward) = action(location, direction);
            let new_value = values[new_state];
            if new_value > best_value {best_value = new_value; best_direction = direction;};
        }
        policy[location] = best_direction;
    }

    policy
}