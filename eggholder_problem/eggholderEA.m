function eggholderEA()
  alpha = 0.05;     % Mutation probability
  lambda = 100;     % Population and offspring size
  k = 3;            % Tournament selection
  intMax = 500;     % Boundary of the domain, not intended to be changed.

  %% Initialize population
  population = intMax * rand(lambda, 2);

  plotPopulation(population, 0)

  return
  for i = 1 : 20
    %% The evolutionary algorithm
    selected = selection(population, k);
    offspring = crossover(selected);
    joinedPopulation = [mutation(offspring, alpha); population];
    population = elimination(joinedPopulation, lambda);

    %% Show progress
    display(sprintf('Mean fitness: %e', mean(objf(population))))

    plotPopulation(population, i)
  end

% Compute the objective function at the vector of (x,y) values.
function [f] = objf(x)
  sas = sqrt(abs(x(:,1)+x(:,2)));
  sad = sqrt(abs(x(:,1)-x(:,2)));
  f = -x(:,2).*sin(sas) - x(:,1).*sin(sad);
end

% Plot the population.
function plotPopulation(population, i)
  x = linspace(0,intMax,500)';
  y = linspace(0,intMax,500)';
  F = -y' .* sin(sqrt(abs(x + y'))) -x .* sin(sqrt(abs(x - y')));

  surfc(x, y, F', 'FaceLighting', 'gouraud', 'EdgeColor', 'none', 'FaceAlpha', 0.95)
  hold on
  for j = 1 : lambda
     plot3(population(j,1), population(j,2), objf(population(j,:))+1e-1, 'rx', 'LineWidth', 2)
     campos([    -3.651359848873533e+02    -4.073576307439369e+02     1.599738318114707e+04]);
     hold all;
  end
  hold off;
  drawnow;

  % Enable this if you want to print the plots to a file
%     title(sprintf('Mean fitness: %e', mean(objf(population))))
%     set(gcf,'color','w');
%     frame = getframe(1);
%     im = frame2im(frame);
%     [imind,cm] = rgb2ind(im,256);
%     imwrite(imind,cm,sprintf('EvolutionaryAlgorithmZZZ%d.png', i),'png');
%     pause;
  % Run the following command in the terminal to create gifs:
  % ffmpeg -framerate 5 -i EvolutionaryAlgorithm%d.png output.gif
end

% Perform k-tournament selection to select pairs of parents.
function [selected] = selection(population, k)
  selected = zeros(2*lambda, 2);
  for ii = 1 : 2*lambda
    ri = randperm(lambda, k);
    [~, mi] = min( objf(population(ri, :)) );
    selected(ii,:) = population(ri(mi),:);
  end
end

% Perform crossover as in the slides.
function [offspring] = crossover(selected)
  weights = 3*rand(lambda,2) - 1;
  offspring = zeros(lambda, 2);
  for ii = 1 : size(offspring,1)
    offspring(ii,1) = min(intMax,max(0,selected(2*ii-1, 1) + weights(ii,1)*(selected(2*ii, 1)-selected(2*ii-1, 1))));
    offspring(ii,2) = min(intMax,max(0,selected(2*ii-1, 2) + weights(ii,2)*(selected(2*ii, 2)-selected(2*ii-1, 2))));
  end
end

% Perform mutation, adding a random Gaussian perturbation.
function [offspring] = mutation(offspring, alpha)
  ii = find(rand(size(offspring,1),1) <= alpha);
  offspring(ii,:) = offspring(ii,:) + 10*randn(length(ii),2);
  offspring(ii,1) = min(intMax, max(0, offspring(ii,1)));
  offspring(ii,2) = min(intMax, max(0, offspring(ii,2)));
end

% Eliminate the unfit candidate solutions.
function [survivors] = elimination(joinedPopulation, keep)
  fvals = objf(joinedPopulation);
  [~, perm] = sort(fvals);
  survivors = joinedPopulation(perm(1:keep),:);
end

end
