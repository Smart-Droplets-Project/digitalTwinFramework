import datetime

import pcse
from pcse.engine import Engine
from pcse.fileinput.pcsefilereader import PCSEFileReader

import pcse_gym
from pcse_gym.envs.winterwheat import WinterWheat
import pcse_gym.utils.defaults as defaults

import ngsildclient
from ngsildclient import Entity, Client, SmartDataModels


def init_cropgym(
    crop_features=defaults.get_default_crop_features(pcse_env=1),
    costs_nitrogen=10,
    reward="DEF",
    nitrogen_levels=7,
    action_multiplier=1.0,
    years=defaults.get_default_train_years(),
    locations=defaults.get_default_location(),
    crop_parameters=None,
    site_parameters=None,
    soil_parameters=None,
    agro_config=None,
    model_config=None,
):

    def get_action_space(nitrogen_levels=7):
        import gymnasium as gym

        space_return = gym.spaces.Discrete(nitrogen_levels)
        return space_return

    action_space = get_action_space(nitrogen_levels=nitrogen_levels)
    env_return = WinterWheat(
        crop_features=crop_features,
        costs_nitrogen=costs_nitrogen,
        years=years,
        locations=locations,
        action_space=action_space,
        action_multiplier=action_multiplier,
        reward=reward,
        crop_parameters=crop_parameters,
        site_parameters=site_parameters,
        soil_parameters=soil_parameters,
        agro_config=agro_config,
        model_config=model_config,
    )

    return env_return


def create_crop(crop_type):
    global N_CROPS
    N_CROPS += 1

    # Load exmaple model and edit relevant data
    crop = Entity.load(SmartDataModels.SmartAgri.Agrifood.AgriCrop)

    crop.id = f"urn:ngsi-ld:AgriCrop:crop-{N_CROPS}-id"
    crop.prop("description", crop_type)
    crop.prop("name", crop_type)
    if crop_type == "wheat":
        alt_name = "Triticum aestivum L."
    else:
        alt_name = ""
    crop.prop("alternateName", f"{alt_name}")

    return crop


# currently naively divide subparcels into equal parts
# TODO need to divide subparcels more accurately
def _get_area_sub_parcels(area: float, n_rows: int):
    # assume rows are straight and segments the parcel
    n_sub_parcels = n_rows + 1
    area_sub_parcels = area / n_rows

    return [area_sub_parcels for _ in range(n_sub_parcels)]


def _get_sub_parcels(crop: Entity, location: tuple, area: float, n_rows: int):
    areas = _get_area_sub_parcels(area, n_rows)
    id_sub_parcels = 0
    sub_parcel_entities = []
    for area in areas:
        sub_parcel_entity = create_parcel(
            crop, location, is_sub_parcel=True, area=area, id_sub_parcels=id_sub_parcels
        )
        sub_parcel_entities.append(sub_parcel_entity)
        id_sub_parcels += id_sub_parcels

    return sub_parcel_entities


def create_parcel(
    crop: Entity,
    location: tuple,  # tuple of lat and lon coordinates
    crop_parameters: str = None,
    site_parameters: str = None,
    soil_parameters: str = None,
    model_config: str = None,
    agro_config: str = None,
    area_parcel: float = 1.0,
    n_rows: int = 0,
    is_sub_parcel: bool = False,
    id_sub_parcel: int = None,
    **kwargs,
):

    # Load model and edit relevant data
    global N_PARCELS
    parcel = Entity.load(SmartDataModels.SmartAgri.Agrifood.AgriParcel)

    # Load planting dates
    planting_date = ...

    if not is_sub_parcel:
        N_PARCELS += 1
    """
    1. URN - Uniform Resource Name
    2. NGSI-LD 
    3. AgriParcel definition from the Smart Data Initiative
    4. Custom ID part, usually a UUID (Universally Unique IDentifier) or some kind of date format
    """
    if not is_sub_parcel:
        parcel.id = f"urn:ngsi-ld:AgriParcel:parcel-{N_PARCELS}-id"
        parcel_dt = init_cropgym(
            crop_parameters=crop_parameters,
            site_parameters=site_parameters,
            soil_parameters=soil_parameters,
            agro_config=agro_config,
            model_config=model_config,
        )
    else:
        parcel.id = (
            f"urn:ngsi-ld:AgriParcel:subparcel-{id_sub_parcel}-id_parcel-{N_PARCELS}-id"
        )
        parcel_dt = None

    # parcel.landLocation = None
    parcel.prop("location", location)
    parcel.prop("area", area_parcel)  # in ha
    parcel.rel("hasAgriCrop", crop.id)
    if n_rows:
        sub_parcels, _ = _get_sub_parcels(crop, location, area_parcel, n_rows)
        parcel.rel("hasAgriParcelChildren", sub_parcels)
    parcel.rel("hasDevices", f"LAI-{N_PARCELS}")
    parcel.rel("hasDevices", f"soilMoisture-{N_PARCELS}")

    parcel.pprint()

    return parcel, parcel_dt


def register_dt_entity(entity):

    with Client() as client:

        print(f"Inserting entity! [{entity.id} {entity.type}]")
        client.create(entity)


def main(): ...


if __name__ == "__main__":
    main()
