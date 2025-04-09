import datetime
import time
import json

from sd_data_adapter.client import DAClient
from sd_data_adapter.models import AgriFood, Devices, AutonomousMobileRobot
from sd_data_adapter.api import search, get_by_id


def get_demo_parcels(description: str = "Lithuania"):
    return {"type": "AgriParcel", "q": f'description=="{description}"'}


def find_parcel_operations(parcel_id: str):
    return search(
        {"type": "AgriParcelOperation", "q": f'hasAgriParcel=="{parcel_id}"'},
        ctx=AgriFood.ctx,
    )


def find_crop(parcel_id: str):
    parcel = get_by_id(parcel_id, ctx=AgriFood.ctx)
    return parcel.hasAgriCrop["object"]


def find_device(crop_id: str):
    return search(
        {"type": "Device", "q": f'controlledAsset=="{crop_id}"'},
        ctx=Devices.ctx,
    )


def find_device_measurement(controlled_property: str = None, ref_device: str = None):
    query = {"type": "DeviceMeasurement"}
    conditions = []

    if controlled_property:
        conditions.append(f'controlledProperty=="{controlled_property}"')
    if ref_device:
        ref_device_str = json.dumps(ref_device)
        conditions.append(f"refDevice=={ref_device_str}")

    if conditions:
        query["q"] = ";".join(conditions)
    return search(query, ctx=Devices.ctx)


def find_command_messages():
    return search(
        {"type": "CommandMessage"},
        ctx=AutonomousMobileRobot.ctx,
    )


def find_agriproducttype(name: str):
    return search(
        {"type": "AgriProductType", "q": f'name=="{name}"'},
        ctx=AgriFood.ctx,
    )


def has_demodata(description: str = "Lithuania"):
    return bool(search(get_demo_parcels(description=description), ctx=AgriFood.ctx))


def clear_database(verbose=False):
    retries = 0
    max_retries = 5  # Set a limit to avoid endless retries
    while retries < max_retries:
        try:
            with DAClient.get_instance() as client:
                client.purge()
            print("Purge completed successfully.")
            return 0  # Success
        except Exception as e:
            retries += 1
            if verbose:
                print(f"Attempt {retries}/{max_retries} failed. Error: {e}")
            if retries < max_retries:
                if verbose:
                    print("Retrying in 3 seconds...")
                time.sleep(3)  # Wait before retrying
            else:
                if verbose:
                    print("Max retries reached. Could not clear the database.")
                return 1  # Failure after max retries


def get_parcel_operation_by_date(parcel_operations, target_date):

    matching_operations = list(
        filter(
            lambda op: datetime.datetime.strptime(op.plannedStartAt, "%Y%m%d").date()
            == target_date,
            parcel_operations,
        )
    )
    return matching_operations[0] if matching_operations else None


def get_matching_device(devices, variable: str):
    matching_devices = list(
        filter(
            lambda op: op.controlledProperty == variable,
            devices,
        )
    )
    return matching_devices[0] if matching_devices else None
