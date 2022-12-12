use plotters::prelude::*;
use ndarray::prelude::*;
use ndarray::Array; 
use std::f64::INFINITY;
use std::f64::NEG_INFINITY;
use std::f64::consts::PI;

const FOURIER_ORDER:usize = 3;

const EPSILON: f64 = 0.05;
const GAMMA: f64 = 1.0;
const LAMBDA: f64 = 0.90;
const ALPHA: f64 = 0.001;

#[derive(Copy, Clone)]
struct State {
    x:f64,
    xdot:f64,
    reward: f64,
    action:usize,
}
struct Model {
    weights: Array<f64, Dim<[usize; 2]>>,
    traces: Array<f64, Dim<[usize; 2]>>,
    gradients: Array<f64, Dim<[usize; 1]>>,
    old_q:f64,
}

fn main() {
    let mut model = Model {
        weights : Array::<f64,_>::zeros((3,FOURIER_ORDER*FOURIER_ORDER).f()),
        traces: Array::<f64,_>::zeros((3,FOURIER_ORDER*FOURIER_ORDER).f()),
        gradients : get_gradient_matrix(),
        old_q:0.0,
    };
    let mut history:Vec<i32> = Vec::new();

    for _i in 0..1000 {
        let mut iters = 0;
        let mut state = init_state();
        state.action = get_action(&state, &model, true);
        model.traces = Array::<f64,_>::zeros((3,FOURIER_ORDER*FOURIER_ORDER).f());

        while state.reward != 0.0 && iters < 5000 {
            let mut new_state = update_state(state);
            new_state.action = get_action(&new_state,&model,true);

            model = update_model(model, &state, &new_state);
            state = new_state;
            iters += 1;
        }
        //println!("Traces: {:?}",model.traces);
        println!("Completed, Total iterations: {}",iters);
        history.push(iters);
        //println!("Weights: {:?}",model.action_weights);

    }
    println!("Weights: {:?}",model.weights);
    println!("{:?}",history);
    let avg_result : f64 = history.iter().fold(0, |acc, x| acc + x) as f64 /history.len() as f64;
    println!("{}",avg_result);
    generate_value_plot(&model);
    print_convergence(history);

}
fn fourier_basis(state: &State) -> Array<f64, Dim<[usize; 1]>>{ 
    //map each feature onto 0,1
    let adjusted_x = (state.x+ 1.2) / 1.7;
    let adjusted_xdot = (state.xdot+0.07)/ 0.14;

    //there is some fancy way to do this with Zip. I do not know it. So much work for replicating np.dot
    let mut data = Array::<f64, _>::zeros((FOURIER_ORDER*FOURIER_ORDER).f());
    for (index,value) in data.iter_mut().enumerate(){
        let i = index / FOURIER_ORDER;
        let j = index % FOURIER_ORDER;
        *value = ((i as f64 * adjusted_x+ j as f64 * adjusted_xdot) * PI).cos();
    }
    data
}

fn fourier_q_value(model:&Model,state: &State) -> f64{
    fourier_basis(&state).dot(&model.weights.slice(s![state.action,..]))
}

fn get_action(state: &State, model: &Model, explore : bool) -> usize {  //I used to like rust
    let mut best_action: usize = 0;
    let mut best_value: f64 =NEG_INFINITY;
    if rand::random::<f64>() < EPSILON && explore{
        best_action = rand::random::<usize>() % 3;
    }
    else {
        for (i,_actions) in model.weights.outer_iter().enumerate() {
            let test_state = State { action:i,..*state};
            let new_value = fourier_q_value(&model, &test_state);
            if new_value > best_value{ 
                best_value = new_value;
                best_action = i;
            }
            //println!("Value for action {} is {}",i,new_value);
        }
    }
    if false && explore {
        println!("Best action is {}",best_action);
    }
    best_action
}

fn update_state(mut state: State) -> State {
    let accel_input:f64 = [-1.0,0.0,1.0][state.action];
    let mut new_xdot = state.xdot + 0.001*accel_input - 0.0025* (3.0*state.x).cos();
    let mut new_x = state.x + new_xdot;
    if new_x <= -1.2 {
        new_x = -1.2;
        new_xdot = 0.0;
    } else if new_x >= 0.5 {
        state.reward = 0.0;
    }
    if new_xdot <= -0.07 {
        new_xdot = -0.07;
    } else if new_xdot >= 0.07 {
        new_xdot = 0.07;
    }
    //println!("X: {}",state.action);
    state.x = new_x;
    state.xdot = new_xdot;
    state
}

fn update_model(mut model: Model, state:&State,new_state:&State) -> Model {
    let q = fourier_q_value(&model, &state);
    let new_q = fourier_q_value(&model, &new_state);
    let current_features = fourier_basis(&state);
    
    //println!("State: {},{}",state.x,state.xdot);
    let delta = state.reward + GAMMA * new_q - q;
    
    
    model.traces *= GAMMA * LAMBDA; //this is technically slightly different than in the book but it shouldn't be relevant and there was a compiler bug when I tried to do it identically
    
    //adds the factor in the true online SARSA example in barto to the trace for the action taken
    let mut action_traces = model.traces.slice_mut(s![state.action,..]);
    let z_x_dot = action_traces.dot(&current_features);
    for (trace,state) in action_traces.iter_mut().zip(current_features.iter()) {
        *trace +=state - ALPHA*z_x_dot*state
    }
    //action_traces += &current_state;
     //it literally will not compile if this is on one line
    let weight_update = ALPHA * &model.gradients*( (delta)*&model.traces);
    model.weights  += &weight_update; 
    
    let mut action_weights = model.weights.slice_mut(s![state.action,..]);
    let action_traces = model.traces.slice(s![state.action,..]);
    for (((weight,feature),trace),gradient) in action_weights.iter_mut().zip(current_features.iter()).zip(action_traces.iter()).zip(model.gradients.iter()) {
        *weight -=ALPHA*gradient * (q -  &model.old_q) * (trace- feature);
    }
    //println!("Traces: {:?}",model.traces);
    //println!("Update: {:?}",weight_update);
    model.old_q = new_q;
    model
}

fn init_state() -> State{
    State {
        x: (rand::random::<f64>() / 5.0)-0.6,   
        xdot: 0.0,
        reward: -1.0, 
        action:1,
    }
}

fn get_gradient_matrix() -> Array<f64, Dim<[usize; 1]>> {
    let mut gradients = Array::<f64,_>::zeros((FOURIER_ORDER*FOURIER_ORDER).f());
    for (index,value) in gradients.iter_mut().enumerate(){
        let i = index / FOURIER_ORDER;
        let j = index % FOURIER_ORDER;
        if index == 0 {
            *value = 1.0;
        }
        else {
        *value = 1.0 / ((i * i + j * j) as f64);
        }
    }
    gradients
}

fn generate_value_plot(model :&Model)-> Result<(), Box<dyn std::error::Error>>{
    //shamelessly cribbed from the plotters github because my god is this a lot of boilerplate
    let xiter = (-15..=15).map(|x| x as f64 / 17.65-0.35);
    let yiter = (-15..=15).map(|x| x as f64 / 214.0);
    let z = |x,y| -fourier_q_value(&model, &State {x:x,xdot:y,reward:0.0,action:get_action(&State {x:x,xdot:y,reward:0.0,action:0}, &model, false)}); //horrifying kludges my beloved

    let mut max = NEG_INFINITY;
    let mut min = INFINITY;
    for x in xiter {
        for y in yiter.clone(){
            let val = z(x,y);
            if val > max{
                max = val;
            }
            if val < min {
                min = val;
            }
        }
    }

    let root = BitMapBackend::new("value.png", (600, 400)).into_drawing_area();
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .caption("Cost Function", ("sans-serif", 30))
            .build_cartesian_3d(-1.2..0.5 as f64, min..max as f64, -0.07..0.07 as f64)?;
        chart.with_projection(|mut p| {
            p.pitch = 0.5;
            p.scale = 0.7;
            p.into_matrix() // build the projection matrix
        });

        chart
            .configure_axes()
            .light_grid_style(BLACK.mix(0.15))
            .max_light_lines(3)
            .draw()?;
        chart.draw_series(
            SurfaceSeries::xoz(
                (-15..=15).map(|x| x as f64 / 17.65-0.35),
                (-15..=15).map(|x| x as f64 / 214.0),
                z,
            )
            .style_func(&|&v    | {
                (&HSLColor((v-min) / (max - min) , 1.0, 0.7)).into()
            }),
        )?;

        root.present()?;

    Ok(())
}

fn print_convergence(history:Vec<i32>) -> Result<(), Box<dyn std::error::Error>>{
    let root = BitMapBackend::new("convergence.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(format!("Convergence for basis of order {}",FOURIER_ORDER), ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(45)
        .build_cartesian_2d(0..30 as i32, 0..3000 as i32)?;

    chart.configure_mesh()
    .disable_x_mesh()
    .disable_y_mesh()
    .x_desc("Episode")
    .y_desc("Length")
    .draw()?;
    chart.draw_series(LineSeries::new(
        (0..).zip(history.iter()).map(|(x, y)| (x, *y)),
        &BLUE,
    ))?;
    root.present()?;

    Ok(())
}
