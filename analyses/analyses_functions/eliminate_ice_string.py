# eliminate_ice_string 

def eliminate_ice_string():
    # create a number of bounding boxes to elinate areas with ice
    # this produces a string that its in the where clause of queries

    boxes = []
    boxes.append([-120.0,50.5,-46.8,80.5]) # huson bay, canada, etc.
    boxes.append([-120.0,50.5,-46.8,80.5]) # huson bay, canada, etc.
    boxes.append([39.5,65.0,-46.8,90]) # arctic except n. atlantic
    boxes.append([15.95,59.02,36.23,66.57]) # North Baltic sea
    boxes.append([-173.7,62.0,-158.4,66.8]) # north beiring sea
    boxes.append([130.5,50.6,-174.2,67.8]) #sea of okhotsk
    boxes.append([3.5,78.1,31.9,85.0]) #north of Salvbard
    boxes.append([-179.8,57.4,-156.5,62.2]) #beiring sea, more southern, because it didn't work
    boxes.append([-44.82,-57.93,-29.05,-50.61]) ## south georgia island
    boxes.append([31.4,61.4,60.3,73.1])## far northeast russia -- a small area
    boxes.append([-27.61,68,-19.47,68.62]) # tiny piece of ice near ne iceland that annoyed me

    eliminated_locations = '''and not
      (
    '''

    for b in boxes:
        min_lon, min_lat, max_lon, max_lat = b
        if min_lon > max_lon:
            bounding_string = \
    '''   (
          (detect_lon > {min_lon} or detect_lon < {max_lon} ) and
          detect_lat> {min_lat} and detect_lat < {max_lat} )
      or'''
        else:
            bounding_string = \
    '''   ( detect_lon > {min_lon} and detect_lon < {max_lon} and detect_lat> {min_lat} and detect_lat < {max_lat} ) or 
'''
        eliminated_locations+=bounding_string.format(min_lon=min_lon,
                                                             max_lon=max_lon,
                                                             max_lat=max_lat,
                                                             min_lat=min_lat)
    eliminated_locations = eliminated_locations[:-4] + ")\n"

    return eliminated_locations
    # print(eliminated_locations)


