import datetime
from collections import defaultdict

from sd_data_adapter.client import DAClient
from sd_data_adapter.models import AgriFood, Devices, AutonomousMobileRobot
from sd_data_adapter.api import search, get_by_id


def get_demo_parcels(description: str = "initial_site"):
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


def count_devices_by_asset():
    # Fetch all devices
    all_devices = search({"type": "Device"}, ctx=Devices.ctx)
    # Dictionary to store count of devices per crop and controlledAsset
    asset_count = defaultdict(lambda: defaultdict(int))
    # Loop through each device
    for device in all_devices:
        # Increment count for this crop and controlledAsset
        asset_count[device.controlledAsset][device.controlledProperty] += 1

    return asset_count


def find_command_messages():
    return search(
        {"type": "CommandMessage"},
        ctx=AutonomousMobileRobot.ctx,
    )


def has_demodata():
    return bool(search(get_demo_parcels()))


def clear_database():
    # clear database with error handling
    try:
        with DAClient.get_instance() as client:
            client.purge()
        print("Purge completed successfully.")
    except Exception as e:
        # Catch any exceptions that occur and print the error message
        print(f"An error occurred while purging the database: {e}")


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
