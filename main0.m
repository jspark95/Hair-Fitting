% Main0
% Database Setting Code

man_listing = dir('../01. Database/SIMS/obj/man_obj/*.obj');
woman_listing = dir('../01. Database/SIMS/obj/woman_obj/*.obj');
addpath('../01. Database/SIMS/obj/man_obj');
addpath('../01. Database/SIMS/obj/man_obj');

for i = 1:length(man_listing)
    hairObj{i} = readObj(man_listing(i).name); % man
end

for i = 1:length(man_listing)
%     hairObj{i} = readObj(man_listing(i).name); % man
    hairObj_v{i} = hairObj{i}.v;
    hairObj_f{i} = hairObj{i}.f.v;
end

save man_hair_v hairObj_v;
 