import pandapower as pp


def main() -> None:
    net = pp.networks.create_cigre_network_mv(with_der="pv_wind")
    net.load["curtailable"] = False
    pp.toolbox.drop_lines(net, [9])
    net.switch.loc[net.switch.name == "S3", "closed"] = True
    net.load.loc[net.load.bus == 7, "p_mw"] = 5
    net.load.loc[net.load.bus == 7, "curtailable"] = True
    pp.to_json(net, "net.json")


if __name__ == "__main__":
    main()
