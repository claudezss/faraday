import pandapower as pp
import pandas as pd


def fill_missing_names(net: pp.pandapowerNet) -> pp.pandapowerNet:
    """Fill missing names for loads, lines, switches, and transformers."""

    net.bus["name"] = net.bus["name"].astype(str)
    net.load["name"] = net.load["name"].astype(str)
    net.sgen["name"] = net.sgen["name"].astype(str)
    net.line["name"] = net.line["name"].astype(str)
    net.trafo["name"] = net.trafo["name"].astype(str)

    # Initialize curtailable column if it doesn't exist
    if "curtailable" not in net.load.columns:
        net.load["curtailable"] = False

    # Fill load names
    for idx, row in net.load.iterrows():
        if pd.isna(row.get("name")) or row.get("name") == "":
            bus_name = (
                net.bus.loc[row.bus, "name"]
                if "name" in net.bus.columns
                else f"Bus_{row.bus}"
            )
            load_type = row.get("type", "load")
            net.load.loc[idx, "name"] = f"Load_{bus_name}_{load_type}_{idx}"

    # Fill gen names
    for idx, row in net.gen.iterrows():
        if pd.isna(row.get("name")) or row.get("name") == "":
            bus_name = (
                net.bus.loc[row.bus, "name"]
                if "name" in net.bus.columns
                else f"Bus_{row.bus}"
            )
            net.gen.loc[idx, "name"] = f"Gen_{bus_name}_{idx}"

    # Fill sgen names
    for idx, row in net.sgen.iterrows():
        if pd.isna(row.get("name")) or row.get("name") == "":
            bus_name = (
                net.bus.loc[row.bus, "name"]
                if "name" in net.bus.columns
                else f"Bus_{row.bus}"
            )
            net.sgen.loc[idx, "name"] = f"SyncGen_{bus_name}_{idx}"

    # Fill line names
    for idx, row in net.line.iterrows():
        if pd.isna(row.get("name")) or row.get("name") == "":
            from_bus = (
                net.bus.loc[row.from_bus, "name"]
                if "name" in net.bus.columns
                else f"Bus_{row.from_bus}"
            )
            to_bus = (
                net.bus.loc[row.to_bus, "name"]
                if "name" in net.bus.columns
                else f"Bus_{row.to_bus}"
            )
            net.line.loc[idx, "name"] = f"Line_{from_bus}_to_{to_bus}_{idx}"

    # Fill transformer names
    if len(net.trafo) > 0:
        for idx, row in net.trafo.iterrows():
            if pd.isna(row.get("name")) or row.get("name") == "":
                hv_bus = (
                    net.bus.loc[row.hv_bus, "name"]
                    if "name" in net.bus.columns
                    else f"Bus_{row.hv_bus}"
                )
                lv_bus = (
                    net.bus.loc[row.lv_bus, "name"]
                    if "name" in net.bus.columns
                    else f"Bus_{row.lv_bus}"
                )
                net.trafo.loc[idx, "name"] = f"Trafo_{hv_bus}_to_{lv_bus}_{idx}"

    # Fill switch names
    if len(net.switch) > 0:
        for idx, row in net.switch.iterrows():
            if pd.isna(row.get("name")) or row.get("name") == "":
                bus_name = (
                    net.bus.loc[row.bus, "name"]
                    if "name" in net.bus.columns
                    else f"Bus_{row.bus}"
                )
                element_type = row.et
                element_idx = row.element
                net.switch.loc[idx, "name"] = (
                    f"Switch_{bus_name}_{element_type}_{element_idx}_{idx}"
                )

    # Fill bus names if missing
    for idx, row in net.bus.iterrows():
        if pd.isna(row.get("name")) or row.get("name") == "":
            net.bus.loc[idx, "name"] = f"Bus_{idx}"

    return net
