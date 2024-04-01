import os

sub_tasks = [
        "(navigate_to_obj, Mug)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Sink)",
        "(put_on, Mug, SinkBasin)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Mug)",
        "(pour, Mug, Sink)",
        "(navigate_to_obj, CoffeeMachine)",
        "(put_in, Mug, CoffeeMachine)",
        "(toggle_on, CoffeeMachine)",
        "(toggle_off, CoffeeMachine)",
        "(pick_up, Mug)",
        "(put_on, Mug, CounterTop)"]

fold = 'data/images/'

for i,tsk in enumerate(sub_tasks):
    print(f"Task: {tsk}")

    fold_new = str(i)+'_'+tsk.replace("(", "").replace(")", "").replace(", ", "__")
    
    # Create the output directory if it doesn't exist
    print(fold + fold_new)
    os.makedirs(fold + fold_new, exist_ok=True)


    # print(fold_new)
