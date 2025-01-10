#!/usr/bin/env python


import os
import time
import uvicorn
import argparse

from time    import sleep
from loguru  import logger
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse



def run(args):

    setup_logs()

    db_booted = False
    while not db_booted:
        try:
            get_db_service(args.db_string)
            db_booted = True
        except:
            time.sleep(2)
            logger.warning("waiting for the database...")



    if args.db_recreate:
        logger.info("recreating database...")
        recreate_db()

    envs = {"INFERENCE_HOST":args.infer_host, "VOLUME":args.volume}


    app = FastAPI(title=__name__)
    app.include_router(routes.remote_app)
    app.include_router(routes.user_app)
    app.include_router(routes.dataset_app)
    app.include_router(routes.task_app)
    app.include_router(routes.image_app)
    app.include_router(routes.resource_app)
    app.include_router(routes.partition_app)


    @app.on_event("startup")
    def startup_event():
        # create all services for the first time...
        get_io_service(args.volume)
        get_manager_service(args.host, envs=envs)
        scheduler_service = get_scheduler_service()
        scheduler_service.start()    

        if testbed:
            for runner in runners.values():
                runner.run_async()



    @app.on_event("shutdown")
    def shutdown_event():
        scheduler_service = get_scheduler_service()
        scheduler_service.stop()
        logger.info("shutdown event...")

        if testbed:
            for runner in runners.values():
                runner.kill()
                while runner.is_alive():
                    sleep(1)         
            

    @app.get("/status")
    async def get_status():
        return PlainTextResponse("OK", status_code=200)


    app_level = 'info'
    port = int(args.host.split(':')[2])
    uvicorn.run(app, port=port, log_level=app_level, host="0.0.0.0")
                

if __name__ == "__main__":

