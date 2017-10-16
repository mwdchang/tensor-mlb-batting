
def _clean(data):
  for x in xrange(len(data)):
    if data[x] == "" and x > 0:
      data[x] = 0


# Parse people
# 0 - id
# 13 - first
# 14 - last
# 15 - given
def processPeople():
  people = {}
  with open("data/People.csv") as f:
    for idx, line in enumerate(f):
      if idx == 0:
        continue
      data = line.split(",")
      # print(data[0], data[13], data[14], data[15])
      people[data[0]] = data[13] + " " + data[14]

  return people


# Parse HoF
# 0 - id
# 6 - Y/N
def processHoF():
  hofs = {}
  with open("data/HoF.csv") as f:
    for idx, line in enumerate(f):
      if idx == 0:
        continue
      data = line.split(",")
      _clean(data)

      id = data[0]
      ballots = float(data[3])
      votes = float(data[5])
      inducted = data[6]
      # print(id, ballots, votes, inducted)

      if id not in hofs:
        hofs[id] = 1
      
      rank = 1
      if inducted == "Y":
        rank = 3
      elif ballots > 0 and (votes / ballots) > 0.20:
        rank = 2
      else:
        rank = 1
      
      if rank > hofs[id]:
        hofs[id] = rank

  return hofs


# Parse Batting stats
# 0 - id
# 6 - AB
# 7 - R
# 8 - H
# 9 - 2B
# 10 - 3B
# 11 - HR
# 12 - RBI
# 13 - SB
# 15 - BB
# 16 - SO
def processBatting():
  players = {}
  with open("data/Batting.csv") as f:
  
    def incr(id, attr, value): 
      if attr in players[id]:
        players[id][attr] = players[id][attr] + value
      else:
        players[id][attr] = value
  
    for idx, line in enumerate(f):
      if idx == 0:
        continue
  
      data = line.split(",")
      for x in xrange(len(data)):
        if data[x] == "" and x > 0:
          data[x] = 0
  
      id  = data[0]
      ab  = int(data[6])
      r   = int(data[7])
      h   = int(data[8])
      h2  = int(data[9])
      h3  = int(data[10])
      hr  = int(data[11])
      rbi = int(data[12])
      sb  = int(data[13])
  
      if id in players:
        pass
      else:
        players[id] = {}
        players[id]["_id"] = id
  
      incr(id, "AB", ab)
      incr(id, "R", r)
      incr(id, "H", h)
      incr(id, "2B", h2)
      incr(id, "3B", h3)
      incr(id, "HR", hr)
      incr(id, "RBI", rbi)
      incr(id, "Y", 1)
    
    # print('Len', len(players), players["jeterde01"])
    """
    for key in players:
      if players[key]["AB"] > 2500:
        print(key, players[key])
    """

  return players


