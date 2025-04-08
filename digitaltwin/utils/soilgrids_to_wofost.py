"""
 Code borrowed from Allard de Wit and Herman Berghuijs
 Wageningen University & Research
 """

import argparse
import os
import sys
import requests
import yaml
import json

import numpy as np
import pandas as pd

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
SOURCE_FOLDER = os.path.dirname(CURRENT_FOLDER)
soil_save_dir = os.path.join(SOURCE_FOLDER, "cropmodel", "configs", "soil")


def default_soilgrid_variables():
    # Define variables that need to be collected for this location
    soil_variables = ["bdod", "clay", "phh2o", "sand", "silt", "soc", "nitrogen"]
    return soil_variables


def default_zs():
    # Define minimum and maximum depths for each SoilGrids soil layer
    zmins = [0, 5, 15, 30, 60]
    zmaxs = [5, 15, 30, 60, 100]
    return zmins, zmaxs


def default_som_content():
    """
    Default soil organic matter content, it is assumed to be 58%
    """

    return 0.58


def default_range_pf_values():
    return [-1.0, 1.0, 1.3, 1.7, 2.0, 2.3, 2.4, 2.7, 3.0, 3.3, 3.7, 4.0, 4.2, 6.0]


def default_pf_field_capacity():
    return 2.0


def default_pf_wilting_point():
    return 4.2


def default_surface_conductivity():
    return 70


def calculate_is_topsoil(zmin, zmax, zmax_topsoil):
    if (zmin <= zmax_topsoil) & (zmax <= zmax_topsoil):
        is_topsoil = True
    else:
        is_topsoil = False
    return is_topsoil


def calculate_van_genuchten(
        df_soilgrids: pd.DataFrame,
        lower_boundary_top_soil: float = 100.,
) -> pd.DataFrame:
    f_C_to_OM = default_som_content()
    pml_to_pct = 0.1

    ptfw = PedotransferFunctionsWosten()
    df_vgp = df_soilgrids.copy()[["latitude", "longitude", "zmin", "zmax", "soc", "phh2o", "nitrogen"]]
    df_vgp["C"] = df_soilgrids.clay.copy()
    df_vgp["D"] = df_soilgrids.bdod.copy()
    df_vgp["S"] = df_soilgrids.silt.copy()
    df_vgp["OM"] = df_soilgrids.soc.copy() * pml_to_pct * f_C_to_OM
    # assume low residual soil moisture content
    df_vgp["theta_r"] = 0.01

    df_vgp["is_topsoil"] = df_vgp.apply(lambda x: calculate_is_topsoil(x.zmin, x.zmax, lower_boundary_top_soil), axis=1)

    df_vgp["alpha"] = df_vgp.apply(lambda x: ptfw.calculate_alpha(x.C, x.D, x.S, x.OM, x.is_topsoil), axis=1)
    df_vgp["k_sat"] = df_vgp.apply(lambda x: ptfw.calculate_k_sat(x.C, x.D, x.S, x.OM, x.is_topsoil), axis=1)
    df_vgp["labda"] = df_vgp.apply(lambda x: ptfw.calculate_lambda(x.C, x.D, x.S, x.OM, x.is_topsoil), axis=1)
    df_vgp["n"] = df_vgp.apply(lambda x: ptfw.calculate_n(x.C, x.D, x.S, x.OM, x.is_topsoil), axis=1)
    df_vgp["theta_s"] = df_vgp.apply(lambda x: ptfw.calculate_theta_s(x.C, x.D, x.S, x.OM, x.is_topsoil), axis=1)

    return df_vgp


def generate_df_soil_input(
        df_vgp: pd.DataFrame,
) -> pd.DataFrame:
    pct_to_frac = 0.01

    df_model_input = pd.DataFrame()
    df_model_input["Thickness"] = df_vgp.zmax.copy() - df_vgp.zmin.copy()
    df_model_input["RHOD"] = df_vgp.D.copy()
    df_model_input["Soil_pH"] = df_vgp.phh2o.copy()
    df_model_input["FSOMI"] = df_vgp.OM.copy() * pct_to_frac
    df_model_input["CNRatioSOMI"] = df_vgp.soc.copy() / df_vgp.nitrogen.copy()

    # assume value for critical air content
    df_model_input["CRAIRC"] = 0.03

    # determine range of pF values
    pFs = default_range_pf_values()

    CONDfromPF_perlayer = []
    SMfromPF_perlayer = []

    for i in range(len(df_vgp)):
        CONDfromPF = []
        SMfromPF = []
        for j, pF in enumerate(pFs):
            r = calculate_soil_moisture_content(pF, df_vgp.alpha.iloc[i], df_vgp.n.iloc[i], df_vgp.theta_r.iloc[i],
                                                df_vgp.theta_s.iloc[i])
            SMfromPF.extend([pF, float(r)])
            r = calculate_log10_hydraulic_conductivity(pF, df_vgp.alpha.iloc[i], df_vgp.labda.iloc[i],
                                                       df_vgp.k_sat.iloc[i], df_vgp.n.iloc[i])
            CONDfromPF.extend([pF, float(r)])
        CONDfromPF_perlayer.append(CONDfromPF)
        SMfromPF_perlayer.append(SMfromPF)

    df_model_input["CONDfromPF"] = CONDfromPF_perlayer
    df_model_input["SMfromPF"] = SMfromPF_perlayer

    return df_model_input


def generate_soil_yaml(
        df_model_input: pd.DataFrame,
        filename: str = None,
) -> str:
    PFFieldCapacity = default_pf_field_capacity()
    PFWiltingPoint = default_pf_wilting_point()
    SurfaceConductivity = default_surface_conductivity()

    nlayers = len(df_model_input)
    Thickness = df_model_input.Thickness.to_list()
    CNRatioSOMI = df_model_input.CNRatioSOMI.to_list()
    CRAIRC = df_model_input.CRAIRC.to_list()
    FSOMI = df_model_input.FSOMI.to_list()
    RHOD = df_model_input.RHOD.to_list()
    Soil_pH = df_model_input.Soil_pH.to_list()
    SMfromPF_perlayer = df_model_input.SMfromPF.to_list()
    CONDfromPF_perlayer = df_model_input.CONDfromPF.to_list()

    # below we generate the header of the soil input file as YAML input structure
    soil_input_yaml = f"""
     RDMSOL: {sum(Thickness)}
     SoilProfileDescription:
         PFWiltingPoint: {PFWiltingPoint}
         PFFieldCapacity: {PFFieldCapacity}
         SurfaceConductivity: {SurfaceConductivity}
         GroundWater: false
         SoilLayers:
     """

    # Here we generate the properties for each soil layer including layer thickness, hydraulic properties,
    # organic matter content, etc.
    for i in range(nlayers):
        s = f"""    - Thickness: {Thickness[i]}
           CNRatioSOMI: {CNRatioSOMI[i]}
           CRAIRC: {CRAIRC[i]}
           FSOMI: {FSOMI[i]}
           RHOD: {RHOD[i]}
           Soil_pH: {Soil_pH[i]}
           SMfromPF: {make_string_table(SMfromPF_perlayer[i])}
           CONDfromPF: {make_string_table(CONDfromPF_perlayer[i])}
     """
        soil_input_yaml += s

    # A SubSoilType needs to be defined. In this case we make the subsoil equal to the properties
    # of the deepest soil layer.
    soil_input_yaml += \
        f"""    SubSoilType:
           CNRatioSOMI: {CNRatioSOMI[-1]}
           CRAIRC: {CRAIRC[-1]}
           FSOMI: {FSOMI[-1]}
           RHOD: {RHOD[-1]}
           Soil_pH: {Soil_pH[-1]}
           Thickness: {Thickness[-1]}
           SMfromPF: {make_string_table(SMfromPF_perlayer[-1])}
           CONDfromPF: {make_string_table(CONDfromPF_perlayer[-1])}
     """

    soil_dict = yaml.safe_load(soil_input_yaml)
    soil_yaml = json.dumps(soil_dict, indent=6)
    return soil_yaml


def dump_soil_yaml(soil_input_yaml, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(soil_input_yaml)


class PedotransferFunctionsWosten():
    """
    Taken from https://github.com/ajwdewit/pcse_notebooks/
    """

    def calculate_van_genuchten_parameters(self, C, D, S, OM, theta_r, topSoil):
        if (OM == 0):
            OM = 0.01
        dict_vg = {}
        dict_vg["alpha"] = self.calculate_alpha(C, D, S, OM, topSoil)
        dict_vg["n"] = self.calculate_n(C, D, S, OM, topSoil)
        dict_vg["lambda"] = self.calculate_lambda(C, D, S, OM, topSoil)
        dict_vg["k_sat"] = self.calculate_k_sat(C, D, S, OM, topSoil)
        dict_vg["theta_r"] = theta_r
        dict_vg["theta_s"] = self.calculate_theta_s(C, D, S, OM, topSoil)
        return dict_vg

    def calculate_alpha(self, C, D, S, OM, topSoil):
        t_alpha = self.calculate_transformed_alpha(C, D, S, OM, topSoil)
        alpha = np.exp(t_alpha)
        return alpha

    def calculate_n(self, C, D, S, OM, topSoil):
        t_n = self.calculate_transformed_n(C, D, S, OM, topSoil)
        n = np.exp(t_n) + 1
        return n

    def calculate_lambda(self, C, D, S, OM, topSoil):
        t_labda = self.calculate_transformed_lambda(C, D, S, OM, topSoil)
        labda = (10 * np.exp(t_labda) - 10) / (1 + np.exp(t_labda))
        return labda

    def calculate_k_sat(self, C, D, S, OM, topSoil):
        t_k_sat = self.calculate_transformed_ksat(C, D, S, OM, topSoil)
        k_sat = np.exp(t_k_sat)
        return k_sat

    def calculate_theta_s(self, C, D, S, OM, topSoil):
        if (topSoil):
            TS = 1.
        else:
            TS = 0.
        theta_s = 0.7919 + 0.001691 * C - 0.29619 * D - 0.000001491 * S * S + 0.0000821 * OM * OM + 0.02427 * (
                    1 / C) + 0.01113 * (1 / S) + \
                  0.01472 * np.log(
            S) - 0.0000733 * OM * C - 0.000619 * D * C - 0.001183 * D * OM - 0.0001664 * topSoil * S
        return theta_s

    def calculate_transformed_alpha(self, C, D, S, OM, topSoil):
        if (topSoil):
            TS = 1.
        else:
            TS = 0.
        t_alpha = -14.96 + 0.03135 * C + 0.0351 * S + 0.646 * OM + 15.29 * D - 0.192 * topSoil - 4.671 * D * D - 0.000781 * C * C - \
                  0.00687 * OM * OM + 0.0449 * (1 / OM) + 0.0663 * np.log(S) + 0.1482 * np.log(
            OM) - 0.04546 * D * S - 0.4852 * D * OM + 0.00673 * topSoil * C
        return t_alpha

    def calculate_transformed_n(self, C, D, S, OM, topSoil):
        if (topSoil):
            TS = 1.
        else:
            TS = 0.
        t_n = -25.23 - 0.02195 * C + 0.0074 * S - 0.1940 * OM + 45.5 * D - 7.24 * D * D + 0.0003658 * C * C + 0.002885 * OM * OM - 12.81 * (
                    1 / D) - \
              0.1524 * (1 / S) - 0.01958 * (1 / OM) - 0.2876 * np.log(S) - 0.0709 * np.log(OM) - 44.6 * np.log(
            D) - 0.02264 * D * C + 0.0896 * D * OM + 0.00718 * topSoil * C
        return t_n

    def calculate_transformed_lambda(self, C, D, S, OM, topSoil):
        if (topSoil):
            TS = 1.
        else:
            TS = 0.
        t_lambda = 0.0202 + 0.0006193 * C * C - 0.001136 * OM * OM - 0.2316 * np.log(
            OM) - 0.03544 * D * C + 0.00283 * D * S + 0.0488 * D * OM;
        return t_lambda

    def calculate_transformed_ksat(self, C, D, S, OM, topSoil):
        if (topSoil):
            TS = 1.
        else:
            TS = 0.
        t_k_sat = 7.755 + 0.0352 * S + 0.93 * topSoil - 0.967 * D * D - 0.000484 * C * C - 0.000322 * S * S + \
                  0.001 * (1 / S) - 0.0748 * (1 / OM) - 0.643 * np.log(S) - 0.01398 * D * C - 0.1673 * D * OM + \
                  0.02986 * topSoil * C - 0.03305 * topSoil * S
        return t_k_sat


def calculate_water_potential_form_pf(pF):
    psi = np.power(10, pF)
    return psi


def calculate_soil_moisture_content(pF, alpha, n, theta_r, theta_s):
    psi = calculate_water_potential_form_pf(pF)
    soil_moisture_content = theta_r + (theta_s - theta_r) / np.power(1 + (np.power(alpha * psi, n)), 1 - 1 / n)
    return soil_moisture_content


def calculate_log10_hydraulic_conductivity(pF, alpha, labda, k_sat, n):
    psi = calculate_water_potential_form_pf(pF)
    m = 1 - 1 / n;
    ah = alpha * psi
    h1 = np.power(1 + np.power(ah, n), m)
    h2 = np.power(ah, n - 1)
    denom = np.power(1 + np.power(ah, n), m * (labda + 2));
    k_h = k_sat * np.power(h1 - h2, 2) / denom
    COND = np.log10(k_h)
    return COND


def make_string_table(XY_table):
    """Converts a list of X,Y pairs into a formatted string table.
    """
    s = "["
    for x, y in zip(XY_table[0::2], XY_table[1::2]):
        s += f"{x:4.1f}, {y:7.4f}, "
    s += "]"
    s = s.replace("  ", " ")
    return s


def generate_soil_file(longitude, latitude):
    """
    Generate a YAML soil file for given longitude and latitude.

    Args:
        longitude (float): Longitude value.
        latitude (float): Latitude value.
    """

    # Get soil data from SoilGrids™ based on longitude and latitude
    soil_data = get_df_soilgrids(lon=longitude, lat=latitude)

    vg_data = calculate_van_genuchten(soil_data)

    df_soil_input = generate_df_soil_input(vg_data)

    soil_yaml = generate_soil_yaml(df_soil_input)

    # Write the soil data to a YAML file
    path_file = os.path.join(soil_save_dir, f"soil_{longitude}_{latitude}.yaml")
    dump_soil_yaml(soil_yaml, path_file)

    print(f"YAML soil file has been created at {path_file}.")


request_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"


def request_soilgrids(lat, lon) -> dict:
    p1 = {"lat": lat, "lon": lon}
    props = {"property": default_soilgrid_variables(), "depth": get_depth_soilgrids()}
    res = requests.get(request_url, params={**p1, **props})
    result = res.json()

    return result


def get_depth_soilgrids() -> list:
    zmins, zmaxs = default_zs()

    depth_name_template = '{zmin}-{zmax}cm'
    depths = []
    for zmin, zmax in zip(zmins, zmaxs):
        depth = depth_name_template.format(zmin=zmin, zmax=zmax)
        depths.append(depth)

    print(f"depths = {depths}")

    return depths


def get_df_soilgrids(lat: float, lon: float) -> pd.DataFrame:
    print(f"getting soilgrids for longitude: {lon} and latitude: {lat}")
    resultd = request_soilgrids(lat, lon)

    check_value_empty(resultd)

    depths = get_depth_soilgrids()
    zmins, zmaxs = default_zs()

    variables = default_soilgrid_variables()

    soild = {}
    soild["latitude"] = []
    soild["longitude"] = []
    soild["zmin"] = []
    soild["zmax"] = []
    for i in range(0, len(depths)):
        soild["zmin"].append(zmins[i])
        soild["zmax"].append(zmaxs[i])
        soild["latitude"].append(lat)
        soild["longitude"].append(lon)
    for i, var in enumerate(variables):
        var_name = resultd['properties']["layers"][i]['name']
        if (var_name in variables):
            soild[var_name] = []
            for j in range(0, len(depths)):
                raw_value = resultd['properties']["layers"][i]["depths"][j]["values"]["mean"]
                d_factor = resultd["properties"]["layers"][i]["unit_measure"]["d_factor"]
                value = raw_value / d_factor
                soild[var_name].append(value)

    df_soilgrids = pd.DataFrame.from_dict(soild)
    return df_soilgrids


def check_value_empty(data, first=True):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "mean" and value is None:
                raise ValueError(f"The key 'mean' has a value of None! Soil data in"
                                 f"given lat/lon might not exist in SoilGrids!"
                                 f"Please retry with a different lat/lon combination.")
            # Recursively check nested dictionaries or lists
            check_value_empty(value)
    elif isinstance(data, list):
        for item in data:
            check_value_empty(item)


def main():
    if len(sys.argv) == 1:
        print("No arguments provided!")
        print("Usage: python generate_soil_file.py <longitude> <latitude>")
        print("Example: python generate_soil_file.py 12.4924 41.8902")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Generate a YAML soil file for a given longitude and latitude.")
    parser.add_argument("longitude", type=float, help="Longitude for the soil data.")
    parser.add_argument("latitude", type=float, help="Latitude for the soil data.")

    args = parser.parse_args()

    # Validate the output directory
    output_dir = os.path.dirname(soil_save_dir)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate the YAML file
    generate_soil_file(args.longitude,
                       args.latitude)


if __name__ == "__main__":
    main()