from diffusers import (
    DDPMScheduler, 
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler, 
    KDPM2DiscreteScheduler, 
    KDPM2AncestralDiscreteScheduler, 
    EulerDiscreteScheduler, 
    EulerAncestralDiscreteScheduler, 
    HeunDiscreteScheduler, 
    LMSDiscreteScheduler, 
    DEISMultistepScheduler, 
    UniPCMultistepScheduler, 
    DDIMScheduler, 
    PNDMScheduler,
)

def scheduler(
    pipeline,
    V_Prediction,
    Karras,
    Rescale_betas_to_zero_SNR,
    SGMUniform,
    Scheduler,
):
    # Handling schedulers
    Prediction_type = "v_prediction" if V_Prediction else "epsilon"
    scheduler_args = {
        "prediction_type": Prediction_type,
        "use_karras_sigmas": Karras,
        "rescale_betas_zero_snr": Rescale_betas_to_zero_SNR
    }
    
    if SGMUniform:
      scheduler_args["timestep_spacing"] = "trailing"
    Scheduler_used = ["", f"{Scheduler} ", "", "", ""]
    Scheduler_used[0] = "V-Prediction " if Prediction_type == "v_prediction" else ""
    Scheduler_used[2] = "Karras " if Karras else ""
    Scheduler_used[3] = "SGMUniform " if SGMUniform else ""
    Scheduler_used[4] = "with zero SNR betas rescaling" if Rescale_betas_to_zero_SNR else ""
    if Scheduler == "DPM++ 2M":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "DPM++ 2M SDE":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, algorithm_type="sde-dpmsolver++", **scheduler_args)
    elif Scheduler == "DPM++ SDE":
        pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "DPM2":
        pipeline.scheduler = KDPM2DiscreteScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "DPM2 a":
        pipeline.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "DDPM":
        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "Euler":
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "Euler a":
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "Heun":
        pipeline.scheduler = HeunDiscreteScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "LMS":
        pipeline.scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "DEIS":
        pipeline.scheduler = DEISMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "UniPC":
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "DDIM":
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "PNDM":
        pipeline.scheduler = PNDMScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
